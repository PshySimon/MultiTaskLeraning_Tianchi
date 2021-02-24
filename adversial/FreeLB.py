"""
@Author:    Pshy Simon
@Date:  2020/12/15 0015 下午 07:18
@Description:
   FreeLB的实现
"""
import torch
from utils.Utils import TASK

adv_init_mag = 0
norm_type = 1
adv_steps = 2
adv_lr = 3
adv_max_norm = 4


class FreeLB:
    # 不同的任务有着不同的参数
    def __init__(self, model):
        self.model = model
        self.tr_loss = 0
        self.config = model.config
        self.params = [self.config.freelb_ocemotion,
                       self.config.freelb_ocnli,
                       self.config.freelb_tnews]

    def process(self, batch, inputs, task):
        # ============================ Code for adversarial training=============
        # initialize delta
        embeds_init = self.model.bert.embeddings.word_embeddings(batch[0])
        input_mask = inputs['attention_mask'].to(embeds_init)
        input_lengths = torch.sum(input_mask, 1)
        # check the shape of the mask here..
        delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
        dims = input_lengths * embeds_init.size(-1)
        mag = self.params[task][adv_init_mag] / torch.sqrt(dims)
        delta = (delta * mag.view(-1, 1, 1)).detach()

        # the main loop
        for astep in range(self.params[task][adv_steps]):
            # (0) forward
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init
            inputs['input_ids'] = None

            output, loss = self.model(**inputs, task=task)
            # (1) backward
            loss = loss / self.params[task][adv_steps]
            self.tr_loss += loss.item()

            loss.backward()

            if astep == self.params[task][adv_steps] - 1:
                # further updates on delta
                break

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + self.params[task][adv_lr] * delta_grad / denorm).detach()
            if self.params[task][adv_max_norm] > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > self.params[task][adv_max_norm]).to(embeds_init)
                reweights = ((self.params[task][adv_max_norm] / delta_norm * exceed_mask
                              + (1 - exceed_mask)).view(-1, 1, 1))
                delta = (delta * reweights).detach()

            embeds_init = self.model.bert.embeddings.word_embeddings(batch[0])
        # ============================ End (2) ==================

    def train(self, **kwargs):
        ocemotion_batch = kwargs['ocemotion_batch']
        ocnli_batch = kwargs['ocnli_batch']
        tnews_batch = kwargs['tnews_batch']
        batch = [ocemotion_batch, ocnli_batch, tnews_batch]
        for task in range(len(TASK)):
            inputs = {"attention_mask": batch[task][1], "token_type_ids":batch[task][2], "label": batch[task][3]}
            self.process(batch[task], inputs, task)