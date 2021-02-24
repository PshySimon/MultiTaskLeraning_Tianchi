"""
损失函数：主要负责对损失函数类型的选择，其他几个类主要负责对收敛速度、loss量级等等
"""
import torch.nn as nn
from losses import FocalLoss, DiceLoss


class Criterion:

    def __init__(self, config, label_weights):
        self.config = config
        self.use_label_weights = self.config.use_label_weights
        self.label_weights = label_weights
        self.loss_weight_strategy = self.config.loss_weight_strategy
        self.uw = UncertaintyWeightLoss(self.config) if self.loss_weight_strategy == 'uw' else None
        

    def get_loss(self, label, pred, task):
        if self.config.loss_function == 'cross_entropy':
            loss = nn.CrossEntropyLoss(weight = self.label_weights[task] if self.label_weights is not None else None)
        elif self.config.loss_function == 'focal_loss':
            if self.label_weights is None:
                raise RuntimeError("标签权重为空")
            loss = FocalLoss.MultiFocalLoss(len(self.label_weights[task]), self.label_weights[task])
        elif self.config.loss_function == 'dice_loss':
            loss = DiceLoss.MultiDiceLoss()
        else:
            raise RuntimeError("没有实现的损失函数")
        return self.uw.get_loss(loss(pred, label), task) if self.uw is not None else loss(pred, label)


    