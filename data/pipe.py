from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key
from tqdm import tqdm


def cmp(v1, v2):
    if v1['from'] == v2['from']:
        return v1['to'] - v2['to']
    return v1['from'] - v2['from']


def cmp_opinion(v1, v2):
    if v1[1]['from'] == v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


class BartACOSPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', opinion_first=True):
        super(BartACOSPipe, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping = {
            'EOQ': '<<eoq>>',  # start of quadruple
            'NONE': '<<none>>',
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>'
        }
        self.opinion_first = opinion_first

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_tokens = cur_num_tokens

    def add_tokens(self):
        tokens_to_add = list(self.mapping.values())
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens


        for tok in tokens_to_add:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id

        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + tokens_to_add
        self.tokenizer.add_tokens(tokens_to_add)
        self.mapping2id = {}
        self.mapping2targetid = {}


        for i, value in enumerate(self.mapping.values()):
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))[0]
            assert key_id == self.cur_num_tokens + i

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid) + 2

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'category': str
            'term': List[str]
        }],
        opinions: [{
            'index': int
            'from': int
            'to': int
            'sentiment': str
            'term': List[str]
        }]

        output:[a_s, a_e, o_s, o_e, c, s]
        :param data_bundle:
        :return:
        """
        self.add_tokens()
        target_shift = len(self.mapping) + 2

        for name in ['train', 'dev', 'test']:
            ds = data_bundle.get_dataset(name)
            acos_ds = DataSet()

            for ins in tqdm(ds, f'preprocessing {name} dataset'):
                raw_words = ins['raw_words']
                word_bpes = [[self.tokenizer.bos_token_id]]
                for word in raw_words:
                    bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                    bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                    word_bpes.append(bpes)
                word_bpes.append([self.tokenizer.eos_token_id])

                lens = list(map(len, word_bpes))
                cum_lens = np.cumsum(list(lens)).tolist()
                _word_bpes = list(chain(*word_bpes))

                acos_target = [0]
                aspects = ins['aspects']
                opinions = ins['opinions']
                for aspect, opinion in zip(aspects, opinions):
                    a_s_bpe = cum_lens[aspect['from']] + target_shift if aspect['from'] != -1 else self.mapping2targetid['NONE']
                    a_e_bpe = cum_lens[aspect['to'] - 1] + target_shift if aspect['to'] != -1 else self.mapping2targetid['NONE']

                    o_s_bpe = cum_lens[opinion['from']] + target_shift if opinion['from'] != -1 else self.mapping2targetid['NONE']
                    o_e_bpe = cum_lens[opinion['to'] - 1] + target_shift if opinion['to'] != -1 else self.mapping2targetid['NONE']

                    category = self.mapping2targetid[aspect['category']]
                    sentiment = self.mapping2targetid[opinion['sentiment']]

                    acos_target += [a_s_bpe, a_e_bpe,
                                    o_s_bpe, o_e_bpe,
                                    category,
                                    sentiment,
                                    self.mapping2targetid['EOQ']]

                acos_target.append(1)
                acos_ins = Instance(src_tokens=_word_bpes.copy(), tgt_tokens=acos_target)
                acos_ds.append(acos_ins)

            data_bundle.set_dataset(acos_ds, name)

        data_bundle.set_pad_val('tgt_tokens', 1)
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:

        with open(paths + '/category.json', 'r', encoding='utf-8') as f:
            catecorys = json.load(f)
            for category in catecorys:
                self.mapping[category] = category.lower()
        data_bundle = ABSALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class ABSALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        delete = 0
        for ins in data:
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']

            new_aspects, new_opinions = [], []
            for aspect, opinion in zip(aspects, opinions):
                if 'category' not in aspect or "sentiment" not in opinion:
                    delete += 1
                    continue
                new_aspects.append(aspect)
                new_opinions.append(opinion)

            ins = Instance(raw_words=tokens, aspects=new_aspects, opinions=new_opinions)
            ds.append(ins)
            if self.demo and len(ds) > 20:
                break
        print(f"For path:{path}, delete {delete} conflicts.")
        return ds


if __name__ == '__main__':
    data_bundle = BartACOSPipe().process_from_file('../datasets/rest_acos')
    print(data_bundle)

