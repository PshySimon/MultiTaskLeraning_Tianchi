"""
@Author:    Pshy Simon
@Date:  2020/12/18 0018 上午 10:45
@Description:
    Dynamic Task Prioritization---动态任务优先级
    论文：Dynamic task prioritization for multitask learning
    ECCV 2018
    主要思想：让更难学的任务具有更高的权重
"""
import torch
from math import log
from sklearn.metrics import accuracy_score


class DynamicTaskPrioritizationLoss:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.y = self.config.dtp_y

    def _calculate_weight(self, kpi, y):
        kpi = max(0.1, kpi)
        kpi = min(0.99, kpi)
        w = -1 * ((1 - kpi) ** y) * log(kpi)
        return w

    def _get_acc(self, out, label):
        return accuracy_score(label.float().detach().cpu().numpy().tolist(),
                              torch.argmax(out, dim=-1).float().detach().cpu().numpy().tolist())

    def get_loss(self, **kwargs):
        tnews_label = kwargs['tnews_label']
        ocnli_label = kwargs['ocnli_label']
        ocemotion_label = kwargs['ocemotion_label']
        tnews_out = kwargs['tnews_out']
        ocnli_out = kwargs['ocnli_out']
        ocemotion_out = kwargs['ocemotion_out']
        tnews_loss = kwargs['tnews_loss']
        ocemotion_loss = kwargs['ocemotion_loss']
        ocnli_loss = kwargs['ocnli_loss']
        
        tnews_acc = self._get_acc(tnews_out, tnews_label)
        ocnli_acc = self._get_acc(ocnli_out, ocnli_label)
        ocemotion_acc = self._get_acc(ocemotion_out, ocemotion_label)

        tnews_kpi = 0.1 if tnews_acc == 0 else tnews_acc
        ocnli_kpi = 0.1 if ocnli_acc == 0 else ocnli_acc
        ocemotion_kpi = 0.1 if ocemotion_acc == 0 else ocemotion_acc

        return ocemotion_loss * self._calculate_weight(ocemotion_kpi, self.y) + ocnli_loss * self._calculate_weight(ocnli_kpi, self.y) + tnews_loss * self._calculate_weight(tnews_kpi, self.y)
