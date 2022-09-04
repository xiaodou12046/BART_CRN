import sys
sys.path.append('../')
import os
import numpy as np
import random
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

import warnings
warnings.filterwarnings('ignore')
from data.pipe import BartACOSPipe
from model.bart import BartSeq2SeqModel, Restricter
import argparse
from fastNLP import Trainer, CrossEntropyLoss, Tester
from model.metrics import ACOSSpanMetric
from model.losses import LossFuction
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results
from model.callbacks import FitlogCallback, WarmupCallback
from fastNLP.core.callback import EarlyStopCallback
from fastNLP.core.sampler import SortedSampler
from model.generater import SequenceGeneratorModel
from model.contrastive import ContrastiveModel

import fitlog
# fitlog.debug()
# fitlog.commit(__file__)
fitlog.set_log_dir('logs')
fitlog.add_hyper_in_file(__file__)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='rest_acos', type=str, choices=['lap_acos', 'rest_acos'])
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--bart_lr', default=3e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--num_workers', default=16, type=int)  # 16
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=1, type=int)
parser.add_argument('--decoder_type', default='avg_feature', type=str, choices=['None', 'avg_score', 'avg_feature'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--warmup', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=58)
parser.add_argument('--save_path', type=str, default='caches/best_model')
parser.add_argument('--early_stop', type=int, default=10)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
print(f'seed: {args.seed}')

lr = args.lr
bart_lr = args.bart_lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
num_workers = args.num_workers
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
warmup = args.warmup
alpha = args.alpha
beta = args.beta
temperature = args.temperature
save_path = f'{args.save_path}/{args.dataset_name}'
print(save_path)
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
fitlog.add_hyper(args)
use_encoder_mlp = args.use_encoder_mlp
early_stop = args.early_stop

# ######hyper#######
demo = False
if demo:
    cache_fn = f"caches/{dataset_name}_{opinion_first}_demo.pt"
else:
    cache_fn = f"caches/{dataset_name}_{opinion_first}.pt"


@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartACOSPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'./datasets/{dataset_name}', demo=demo)

    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid


data_bundle, tokenizer, mapping2id, mapping2targetid = get_data()
conflict_id = -1 if 'CON' not in mapping2targetid else mapping2targetid['CON']
print(data_bundle)
max_len = 40
max_len_a = {
    'lap_acos': 0.6,
    'rest_acos': 0.6
}[dataset_name]

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0
eos_token_id = 1
label_ids = list(mapping2id.values())


seq2seq_model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)
contrastive_model = ContrastiveModel(seq2seq_model.encoder, seq2seq_model.decoder.decoder,
                                     input_dim=seq2seq_model.encoder.bart_encoder.embed_tokens.weight.shape[-1],
                                     output_dim=2, label_ids=label_ids, tokenizer=tokenizer, device=device)

vocab_size = len(tokenizer)
print(vocab_size, seq2seq_model.decoder.decoder.embed_tokens.weight.data.size(0))
restricter = Restricter(label_ids)

model = SequenceGeneratorModel(seq2seq_model, contrastive_model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id, max_length=max_len, max_len_a=max_len_a,
                               num_beams=num_beams, do_sample=False, repetition_penalty=1,
                               length_penalty=length_penalty, pad_token_id=eos_token_id, restricter=None)

parameters = []
params = {'lr': lr, 'weight_decay': 1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if not ('encoder' in name or 'decoder' in name) or ('encoder_mlp' in name) or ('gcn' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr': bart_lr, 'weight_decay': 1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('encoder' in name or 'decoder' in name) and not ('layernorm' in name or 'layer_norm' in name) and not ('encoder_mlp' in name) and not ('gcn' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr': bart_lr, 'weight_decay': 0}
params['params'] = []
for name, param in model.named_parameters():
    if ('encoder' in name or 'decoder' in name) and ('layernorm' in name or 'layer_norm' in name) and not ('encoder_mlp' in name) and not ('gcn' in name):
        params['params'].append(param)
parameters.append(params)
optimizer = optim.AdamW(parameters)

num_params = 0
for param in model.parameters():
    num_params += param.numel()
print(f'number of parameters:{num_params}')

callbacks = []
callbacks.append(EarlyStopCallback(early_stop))
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=warmup, schedule='linear'))
callbacks.append(FitlogCallback(tester={
    'test': Tester(data=data_bundle.get_dataset('test'), model=model,
                    metrics=ACOSSpanMetric(eos_token_id, num_labels=len(label_ids), contrastive_model=contrastive_model, device=device),
                    batch_size=batch_size, num_workers=num_workers, device=device, verbose=0, use_tqdm=False, fp16=False)}))


train_data = data_bundle.get_dataset('train')
dev_data = data_bundle.get_dataset('dev')

sampler = BucketSampler(seq_len_field_name='src_seq_len', batch_size=batch_size)
metric = [ACOSSpanMetric(eos_token_id, num_labels=len(label_ids), contrastive_model=contrastive_model, device=device)]

trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer,
                  loss=LossFuction(alpha, beta, temperature, device),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=num_workers, n_epochs=n_epochs, print_every=1 if 'SEARCH_OUTPUT_FP' not in os.environ else 100,
                  dev_data=dev_data, metrics=metric, metric_key='gen_f',
                  validate_every=-1, save_path=save_path, use_tqdm='SEARCH_ID' not in os.environ, device=device,
                  callbacks=callbacks, check_code_level=-1 if 'SEARCH_ID' in os.environ else 0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size)

trainer.train(load_best_model=False)

if trainer.save_path is not None:
    model_name = "best_" + "_".join([model.__class__.__name__, trainer.metric_key, trainer.start_time])
    fitlog.add_other(name='model_name', value=model_name)


