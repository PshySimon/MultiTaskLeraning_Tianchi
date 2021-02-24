"""
@Author:    Pshy Simon
@Date:  2020/12/18 0018 上午 11:15
@Description:
   Dynamic Weight Averaging---动态加权平均
   论文：End-to-End Multi-Task Learning with Attention
   CVPR 2019
   主要思想：希望各个任务以相近的速度来学习
"""
import torch
from utils.Utils import TASK, OCEMOTION, OCNLI, TNEWS


class DynamicWeightAveragingLoss:
    def __init__(self, model):
        self.model = model
        self.record_losses = []
        self.dwa_weights = torch.tensor([1.] * len(TASK)).to(model.config.device)

    def get_loss(self, **kwargs):
        ocemotion_loss = kwargs['ocemotion_loss']
        ocnli_loss = kwargs['ocnli_loss']
        tnews_loss = kwargs['tnews_loss']
        step = kwargs['step']
        self.record_losses.append([ocemotion_loss.item(), ocnli_loss.item(), tnews_loss.item()])
        # 从第二步开始才开始对各个任务限制权重
        if step >= 2:
            L_t_1 = self.record_losses[step - 1]
            L_t_2 = self.record_losses[step - 2]
            rate = [x / y / self.model.config.dwa_T for x, y in zip(L_t_1, L_t_2)]
            self.dwa_weights = len(TASK) * torch.softmax(
                torch.tensor(rate), dim=-1).to(self.model.config.device)
        return (self.dwa_weights[OCEMOTION] * ocemotion_loss +
                self.dwa_weights[OCNLI] * ocnli_loss + self.dwa_weights[TNEWS] * tnews_loss)
