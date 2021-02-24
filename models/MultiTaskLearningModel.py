"""
@Author:    Pshy Simon
@Date:  2020/12/15 0015 下午 04:52
@Description:
    多任务学习模型，三个任务交替训练，得到的loss相加
    采用的是硬共享方式：即公用同一个预训练语言模型，输出层接入不同的层做微调
"""

import torch
from abc import ABC
from math import log
import torch.nn as nn
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertConfig
from utils.Utils import TASK, OCEMOTION, OCNLI, TNEWS
from losses.Criterion import Criterion
from models import SentimentAnalysisModel, TextClassificationModel, NaturalLanguageInferenceModel


class Model(nn.Module, ABC):
    def __init__(self, config, label_weights=None):
        super().__init__()
        self.config = config
        self.label_weights = [torch.tensor(x).to(self.config.device) for x in label_weights] if label_weights is not None else None
        self.bert_config = BertConfig.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout)
        # 三个任务接不同的dense层
        self.dense_tnews = TextClassificationModel.TextRCNN(self.config, self.bert_config)
        self.dense_emotion = SentimentAnalysisModel.RNNAttention(self.config, self.bert_config)
        self.dense_nli = NaturalLanguageInferenceModel.AttentionMLP(self.config, self.bert_config)


    def forward(self, input_ids, attention_mask, token_type_ids, label=None, task=OCEMOTION, inputs_embeds=None):

        if inputs_embeds is None:
            output = self.bert(input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, output_hidden_states=True)
        else:
            output = inputs_embeds

        # 指定任务类型
        if task == TNEWS:
            out = self.dense_tnews(output)
        elif task == OCNLI:
            out = self.dense_nli(output)
        else:
            out = self.dense_emotion(output)
        if label is not None:
            loss = self._get_loss(label, out, task)
        else:
            loss = None
        return tuple(x for x in (out, loss) if x is not None)

    def _get_loss(self, labels, outputs, task):
        criterion = Criterion(self.config, self.label_weights)
        return criterion.get_loss(labels, outputs, task)
    
    def compute_loss(self, ocemotion_loss, ocnli_loss, tnews_loss):
        loss = [ocemotion_loss, ocnli_loss, tnews_loss]
        return sum(loss) if not self.config.use_pcgrad else loss

