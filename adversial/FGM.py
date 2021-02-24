"""
@Author:    Pshy Simon
@Date:  2020/12/15 0015 下午 04:31
@Description:
    对抗验证：FGM
"""
import torch
from utils.Utils import OCEMOTION, OCNLI, TNEWS


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.config = model.config
        self.epsilon = self.config.fgm_epsilon
        self.embedding_name = self.config.fgm_embedding_name

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
    def train(self, **kwargs):
        ocemotion_batch = kwargs['ocemotion_batch']
        ocnli_batch = kwargs['ocnli_batch']
        tnews_batch = kwargs['tnews_batch']
        self.attack()
        for task_batch, task in zip((ocemotion_batch, ocnli_batch, tnews_batch), (OCEMOTION, OCNLI, TNEWS)):
            out_adv, loss_adv = self.model(*task_batch, task)
            loss_adv.backward()
        self.restore()
