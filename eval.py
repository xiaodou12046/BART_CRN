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
import argparse
warnings.filterwarnings('ignore')
from data.pipe import BartACOSPipe
from model.metrics import ACOSSpanMetric
from fastNLP import Tester
from fastNLP import cache_results


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='rest_acos', type=str, choices=['lap_acos', 'rest_acos'])
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--seed', type=int, default=58)
parser.add_argument('--model_path', type=str, default='')
args = parser.parse_args()

# ######random seed######
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

dataset_name = args.dataset_name
batch_size = args.batch_size
num_workers = args.num_workers
opinion_first = args.opinion_first
bart_name = args.bart_name
model_path = args.model_path


# ######hyper#######
@cache_results(f"caches/{dataset_name}_{opinion_first}.pt", _refresh=False)
def get_data():
    pipe = BartACOSPipe(tokenizer=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'./datasets/{dataset_name}', demo=False)

    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid


data_bundle, tokenizer, mapping2id, mapping2targetid = get_data()
print(data_bundle)
print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0
eos_token_id = 1
label_ids = list(mapping2id.values())

# ######load model######
model = torch.load(model_path)
seq2seq_model, contrastive_model = model.seq2seq_model, model.contrastive_model

# ######test######
test_data = data_bundle.get_dataset('test')
metrics = [ACOSSpanMetric(eos_token_id, num_labels=len(label_ids),
                         contrastive_model=contrastive_model, device=device)]

tester = Tester(data=test_data, model=model, num_workers=num_workers,
                metrics=metrics, batch_size=batch_size, device=device,
                verbose=0, use_tqdm=True, fp16=False)
eval_results = tester.test()

print(eval_results)
