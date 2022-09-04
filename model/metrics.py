import torch
from sklearn import metrics
from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from collections import Counter


class ACOSSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, contrastive_model, device):
        super(ACOSSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.word_start_index = num_labels + 2
        self.contrastive_model = contrastive_model
        self.device=torch.device(device)

        self.gen_tp = 0
        self.gen_fp = 0
        self.gen_fn = 0

        self.gen_total = 0
        self.gen_invalid = 0

        self.cont_preds = []
        self.cont_labels = []

    def evaluate(self, pred, tgt_tokens, src_tokens, src_seq_len):
        # self.gen_total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len-2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len-2).tolist()

        for i, (ps, ts) in enumerate(zip(pred.tolist(), tgt_tokens)):
            preds = ps[:pred_seq_len[i]]
            targets = ts[:target_seq_len[i]].detach().cpu().numpy().tolist()

            target_pairs = []
            cur_target_pair = []
            for g in targets:
                if g == 2:
                    target_pairs.append(tuple(cur_target_pair))
                    cur_target_pair = []
                else:
                    cur_target_pair.append(g)

            invalid = 0
            pred_pairs = []
            cur_pred_pair = []
            if len(preds):
                for p in preds:
                    if p == 2:
                        self.gen_total += 1
                        if len(cur_pred_pair) != 6 or cur_pred_pair[0] > cur_pred_pair[1] \
                                or cur_pred_pair[2] > cur_pred_pair[3] \
                                or cur_pred_pair[4] >= self.word_start_index \
                                or cur_pred_pair[5] >= self.word_start_index:
                            invalid += 1
                        else:

                            pred_quad = torch.tensor([cur_pred_pair + [2, 1]]).to(self.device)
                            logit = self.contrastive_model(src_tokens[i:i+1, :],
                                                           src_seq_len[i:i+1],
                                                           pred_quad)
                            if logit[0] == 1:
                                pred_pairs.append(tuple(cur_pred_pair))
                            elif logit[0] == 0:
                                invalid += 1
                        cur_pred_pair = []
                    else:
                        cur_pred_pair.append(p)

            self.gen_invalid += invalid

            acos_target_counter = Counter()
            acos_pred_counter = Counter()

            for t in target_pairs:
                acos_target_counter[(t[0], t[1], t[2], t[3])] = (t[4], t[5])

            for p in pred_pairs:
                acos_pred_counter[(p[0], p[1], p[2], p[3])] = (p[4], p[5])


            gen_tp, gen_fn, gen_fp = _compute_tp_fn_fp([(key[0], key[1], key[2], key[3], value[0], value[1]) for key, value in acos_pred_counter.items()],
                                           [(key[0], key[1], key[2], key[3], value[0], value[1]) for key, value in acos_target_counter.items()])
            self.gen_fn += gen_fn
            self.gen_fp += gen_fp
            self.gen_tp += gen_tp

        contrastive_preds = self.contrastive_model(src_tokens, src_seq_len, tgt_tokens)
        self.cont_preds += contrastive_preds.cpu().numpy().tolist()
        self.cont_labels += [1] * len(contrastive_preds)


    def get_metric(self, reset=True):
        res = {}
        gen_f, gen_pre, gen_rec = _compute_f_pre_rec(1, self.gen_tp, self.gen_fn, self.gen_fp)
        cont_acc, cont_f1 = _conpute_acc_f1(self.cont_preds, self.cont_labels)

        res['gen_f'] = round(gen_f * 100, 2)
        res['gen_rec'] = round(gen_rec * 100, 2)
        res['gen_pre'] = round(gen_pre * 100, 2)
        res['gen_inv'] = round(self.gen_invalid / (self.gen_total + 1e-9), 4)
        res['cont_acc'] = round(cont_acc * 100, 2)
        res['cont_f1'] = round(cont_f1 * 100, 2)

        if reset:
            self.gen_fp = 0
            self.gen_tp = 0
            self.gen_fn = 0
            self.cont_preds = []
            self.cont_labels = []

        return res


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, (list, set)):
        ts = {key:1 for key in list(ts)}
    if isinstance(ps, (list, set)):
        ps = {key:1 for key in list(ps)}
    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp


def _conpute_acc_f1(ps, ts):
    f1 = metrics.f1_score(ps, ts, average='macro', labels=[0, 1])

    correct = 0
    for p, t in zip(ps, ts):
        if p == t:
            correct += 1
    acc = correct / len(ps)

    return acc, f1

