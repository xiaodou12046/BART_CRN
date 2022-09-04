from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
from model.contrastive import SupervisedContrastiveLoss


class LossFuction(LossBase):
    def __init__(self, alpha, beta, temperature, device):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.sup_cont_loss = SupervisedContrastiveLoss(temperature, device)

    def get_loss(self, tgt_tokens, tgt_seq_len, pred, sup_cont_feats, imp_label, bin_pred, bin_label):
        """
        :param tgt_tokens: bsz x max_len, including [sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        mask[:, :1].fill_(1)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)

        generate_loss = F.cross_entropy(pred.transpose(1, 2), tgt_tokens)
        pred_cont_loss = F.cross_entropy(bin_pred, bin_label)
        sup_cont_loss = self.sup_cont_loss(sup_cont_feats, imp_label)

        loss = generate_loss + self.alpha * pred_cont_loss + self.beta * sup_cont_loss
        return loss


