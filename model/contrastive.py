import torch
import random
from copy import copy
import torch.nn as nn
import torch.nn.functional as F
from model.bart import BartState
from torch.utils.data import Dataset, DataLoader, TensorDataset


def get_long_tensor(tokens_list, seq_lens, pad_id):
    # pading tokens
    batch_size = len(seq_lens)
    token_len = max(seq_lens)
    tokens = torch.LongTensor(batch_size, token_len).fill_(pad_id)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


class ContrastiveModel(nn.Module):
    def __init__(self, bart_encoder, bart_decoder, input_dim, output_dim, label_ids, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_dim = input_dim
        self.device = torch.device(device)

        self.pad_token_id = tokenizer.pad_token_id
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids) + 1
        self.src_start_index = len(label_ids) + 2
        self.register_buffer('mapping', torch.LongTensor([0, 2] + label_ids))
        self.register_buffer('causal_masks', torch.zeros(512, 512).fill_(float('-inf')).triu(diagonal=1).float())

        self.encoder = bart_encoder
        self.decoder = bart_decoder
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim),
                                 nn.ReLU(),
                                 nn.Linear(input_dim, output_dim))

        start_tokens = self.tokenizer.tokenize('retrospect the quad', add_prefix_space=True)
        self.start_tokens = self.tokenizer.convert_tokens_to_ids(start_tokens)
        end_tokens = self.tokenizer.tokenize('.', add_prefix_space=True)
        self.end_tokens = self.tokenizer.convert_tokens_to_ids(end_tokens)

    def cont_data_generater(self, src_tokens, src_seq_len, quad_tokens):
        batch_size = src_tokens.size(0)

        mapping_token_mask = quad_tokens.lt(self.src_start_index)
        mapped_tokens = quad_tokens.masked_fill(quad_tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        tgt_tokens_index = quad_tokens - self.src_start_index  # bsz x num_src_token
        tgt_tokens_index = tgt_tokens_index.masked_fill(tgt_tokens_index.lt(0), 0)

        word_mapped_tokens = src_tokens.gather(index=tgt_tokens_index, dim=1)

        quad_input_tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)

        implicit_labels = []
        pos_encoder_input_tokens = []
        pos_decoder_input_tokens = []
        neg_encoder_input_tokens = []
        neg_decoder_input_tokens = []
        EOQ_tokenid = self.mapping[2].cpu().numpy()
        NONE_tokenid = self.mapping[3].cpu().numpy()
        for i, (src, quad) in enumerate(zip(src_tokens.cpu().numpy().tolist(), quad_input_tokens.cpu().numpy().tolist())):
            quad_pairs = []
            cur_quad_pair = []
            for t in quad[1:]:
                if t == EOQ_tokenid:
                    quad_pairs.append(cur_quad_pair)
                    cur_quad_pair = []
                else:
                    cur_quad_pair.append(t)

            for quad_pair in quad_pairs:

                pos_tokens = src[:src_seq_len[i]] + [2] + self.start_tokens + quad_pair + self.end_tokens + [2]
                pos_encoder_input_tokens.append(pos_tokens)
                pos_decoder_input_tokens.append(pos_tokens[:-1])

                if self.training:

                    if quad_pair[0] != NONE_tokenid and quad_pair[2] != NONE_tokenid:
                        implicit_labels.append(0)
                    elif quad_pair[0] == NONE_tokenid and quad_pair[2] != NONE_tokenid:
                        implicit_labels.append(1)
                    elif quad_pair[0] != NONE_tokenid and quad_pair[2] == NONE_tokenid:
                        implicit_labels.append(2)
                    elif quad_pair[0] == NONE_tokenid and quad_pair[2] == NONE_tokenid:
                        implicit_labels.append(3)


                    negtive_tgt_pair = [0] * 6
                    while True:

                        type = random.randint(0, 1)
                        start_idx = random.randrange(1, src_seq_len[i])
                        end_idx = random.randrange(start_idx, src_seq_len[i])
                        if type == 0:
                            negtive_tgt_pair[0], negtive_tgt_pair[1] = src[:src_seq_len[i]][start_idx], src[:src_seq_len[i]][end_idx]
                            negtive_tgt_pair[2:] = quad_pair[2:]
                        else:
                            negtive_tgt_pair[:2] = quad_pair[:2]
                            negtive_tgt_pair[2], negtive_tgt_pair[3] = src[:src_seq_len[i]][start_idx], src[:src_seq_len[i]][end_idx]
                            negtive_tgt_pair[4:] = quad_pair[4:]

                        if negtive_tgt_pair not in quad_pairs:
                            neg_tokens = src[:src_seq_len[i]] + [2] + self.start_tokens + negtive_tgt_pair + self.end_tokens + [2]
                            neg_encoder_input_tokens.append(neg_tokens)
                            neg_decoder_input_tokens.append(neg_tokens[:-1])
                            break

        if self.training:
            sample_idx = random.sample(range(len(neg_encoder_input_tokens)), batch_size)

            pos_encoder_input_tokens = [pos_encoder_input_tokens[idx] for idx in sample_idx]
            pos_decoder_input_tokens = [pos_decoder_input_tokens[idx] for idx in sample_idx]
            neg_encoder_input_tokens = [neg_encoder_input_tokens[idx] for idx in sample_idx]
            neg_decoder_input_tokens = [neg_decoder_input_tokens[idx] for idx in sample_idx]
            implicit_labels = [implicit_labels[idx] for idx in sample_idx]

            encoder_input_tokens = pos_encoder_input_tokens + neg_encoder_input_tokens
            decoder_input_tokens = pos_decoder_input_tokens + neg_decoder_input_tokens

        else:
            encoder_input_tokens = pos_encoder_input_tokens
            decoder_input_tokens = pos_decoder_input_tokens

        encoder_seq_len = list(map(lambda x: len(x), encoder_input_tokens))
        decoder_seq_len = list(map(lambda x: len(x), decoder_input_tokens))

        encoder_input_tokens = get_long_tensor(encoder_input_tokens, encoder_seq_len, self.pad_token_id).to(self.device)
        decoder_input_tokens = get_long_tensor(decoder_input_tokens, decoder_seq_len, self.pad_token_id).to(self.device)
        encoder_seq_len = torch.LongTensor(encoder_seq_len).to(self.device)
        decoder_seq_len = torch.LongTensor(decoder_seq_len).to(self.device)

        if self.training:
            assert len(implicit_labels) == len(pos_encoder_input_tokens), "the number of labels does not match the batch size"
            # assert len(implicit_labels) == encoder_seq_len.size(0) // 2

            implicit_labels = torch.LongTensor(implicit_labels).to(self.device)
            bin_labels = [1] * len(pos_encoder_input_tokens) + [0] * len(neg_encoder_input_tokens)
            bin_labels = torch.LongTensor(bin_labels).to(self.device)
            pos_nums = len(pos_encoder_input_tokens)

            return encoder_input_tokens, decoder_input_tokens, encoder_seq_len, decoder_seq_len, implicit_labels, bin_labels, pos_nums
        else:
            return encoder_input_tokens, decoder_input_tokens, encoder_seq_len, decoder_seq_len

    def prepare_state(self, input_tokens, src_seq_len=None, first=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(input_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, input_tokens, first, src_embed_outputs)
        return state

    def forward(self, src_tokens, src_seq_len, quad_tokens):

        if self.training:
            encoder_input_tokens, decoder_input_tokens, encoder_seq_len, decoder_seq_len, implicit_labels, bin_labels, pos_nums = self.cont_data_generater(src_tokens, src_seq_len, quad_tokens)
        else:
            encoder_input_tokens, decoder_input_tokens, encoder_seq_len, decoder_seq_len = self.cont_data_generater(src_tokens, src_seq_len, quad_tokens)

        state = self.prepare_state(encoder_input_tokens, encoder_seq_len)

        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask
        decoder_pad_mask = decoder_input_tokens.eq(self.pad_token_id)

        dict = self.decoder(input_ids=decoder_input_tokens,
                            encoder_hidden_states=encoder_outputs,
                            encoder_padding_mask=encoder_pad_mask,
                            decoder_padding_mask=decoder_pad_mask,
                            decoder_causal_mask=self.causal_masks[:decoder_input_tokens.size(1), :decoder_input_tokens.size(1)],
                            return_dict=True)

        decoder_output = dict.last_hidden_state


        batch_size = decoder_output.size(0)
        max_seq_len = decoder_output.size(1)
        idx = decoder_seq_len + torch.arange(0, batch_size * max_seq_len, max_seq_len).to(self.device) - 1
        eos_token_embs = decoder_output.reshape(-1, self.input_dim)[idx, :]


        logits = self.mlp(eos_token_embs)

        if self.training:
            sup_cont_feats = eos_token_embs[:pos_nums, ...]
            return sup_cont_feats, logits, implicit_labels, bin_labels
        else:
            pred = torch.max(logits, 1)[1]
            return pred


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, device=None):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, labels):
        batch_size = features.size(0)

        labels = labels.reshape(-1, 1)

        diagonal = torch.eye(batch_size, dtype=torch.bool).to(self.device)

        mask = torch.ones(batch_size, batch_size, dtype=torch.float32).to(self.device)
        mask = mask.masked_fill(diagonal, value=torch.tensor(-1e9))
        # mask self contrastive
        sim_matrix = torch.matmul(features, features.T) * mask / self.temperature  # [batch_size, batch_size]

        sim_matrix = F.softmax(sim_matrix, dim=-1)
        log_sim_matrix = torch.log(sim_matrix + 1e-9)

        sim_mask = (torch.eq(labels, labels.T) * ~diagonal).float()
        mean_log_prob = (sim_mask * log_sim_matrix).sum(-1) / (sim_mask.sum(-1) + 1e-9)

        sup_cont_loss = - mean_log_prob.mean()
        return sup_cont_loss



