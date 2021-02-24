"""
@Author:    Pshy Simon
@Date:  2020/12/18 0018 上午 10:18
@Description:
    Uncertainty Weighting---不确定性加权
    论文：Multi-task learning using uncertainty to weigh losses for scene geometry and semantics
    CVPR 2018
    主要思想：让简单的任务具有更高的权重
"""
import torch
import torch.nn as nn
from utils.Utils import OCEMOTION, OCNLI


class UncertaintyWeightLoss:

    def __init__(self, config):
        self.config = config
        self.log_var_1 = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_2 = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_3 = nn.Parameter(torch.zeros((1,), requires_grad=True))
        model.register_parameter(name='uncertainty1', param=self.log_var_1)
        model.register_parameter(name='uncertainty2', param=self.log_var_2)
        model.register_parameter(name='uncertainty3', param=self.log_var_3)

    def get_loss(self, raw_loss, task):
        if task == OCEMOTION:
            precision = torch.exp(-self.log_var_1).to(self.config.device)
            loss = torch.sum(precision * raw_loss + self.log_var_1)
        elif task == OCNLI:
            precision = torch.exp(-self.log_var_2).to(self.config.device)
            loss = torch.sum(precision * raw_loss + self.log_var_2)
        else:
            precision = torch.exp(-self.log_var_3).to(self.config.device)
            loss = torch.sum(precision * raw_loss + self.log_var_3)
        return loss
