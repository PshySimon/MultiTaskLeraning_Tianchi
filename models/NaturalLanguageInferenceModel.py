"""
@Author:    Pshy Simon
@Date:  2020/12/16 0016 上午 10:24
@Description:
    自然语言推理所要用的模型：
        attention + mlp：比较通用的attention + mlp
        last_4_hidden：选取最后四层作为语义表示
        pooled_output：使用池化层
        12_layers + multi sample dropout：对12层模型加权然后取用multi sample output
"""

import torch
import torch.nn as nn
from abc import ABC
from typing import Tuple


class AttentionMLP(nn.Module, ABC):
    """
    使用Attention + MLP做分类任务
    """

    def __init__(self, config, bert_config):
        super().__init__()
        self.config = config
        self.bert_config = bert_config
        self.atten_layer = nn.Linear(self.bert_config.hidden_size,
                                     self.config.attmlp_head)
        self.dropout = nn.Dropout(self.config.dropout)
        self.softmax_d1 = nn.Softmax(dim=1)
        # 三个任务接不同的dense层
        self.dense_nli = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.bert_config.hidden_size,
                      config.attmlp_head * config.ocnli['num_classes']),
            nn.ReLU(inplace=True),
        )

    def forward(self, embeddings):
        if isinstance(embeddings, Tuple):
            embedded = embeddings[0]
        else:
            embedded = embeddings
        cls_emb = embedded[:, 0, :].squeeze(1)
        # attention_score = [batch_size, attention_heads]
        attention_score = self.atten_layer(cls_emb)
        # attention_score = [batch_size, 1, attention_heads]
        attention_score = self.dropout(
            self.softmax_d1(attention_score).unsqueeze(1))
        out = self.dense_nli(cls_emb).contiguous().view(
            -1, self.config.attmlp_head, self.config.ocnli['num_classes'])
        out = torch.matmul(attention_score, out).squeeze(1)
        return out


class MultiSampleDropout(nn.Module, ABC):
    """
        网络自动选择各层的权重
    """

    def __init__(self, config, bert_config):
        super().__init__()
        self.config = config
        self.bert_config = bert_config
        self.dropout = nn.Dropout(0.2)
        self.high_dropout = nn.Dropout(0.5)

        n_weights = self.bert_config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.classifier = nn.Linear(self.bert_config.hidden_size,
                                    self.config.ocnli['num_classes'])

    def forward(self, embeddings):
        hidden_layers = embeddings[2]
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2)
        cls_output = (torch.softmax(self.layer_weights, dim=0) *
                      cls_outputs).sum(-1)
        out = torch.mean(
            torch.stack(
                [
                    self.classifier(self.high_dropout(cls_output))
                    for _ in range(5)
                ],
                dim=0,
            ),
            dim=0,
        )
        return out


class EmbeddingStacking(nn.Module):
    """
        使用最后四层来做语义表示
    """

    def __init__(self, config, bert_config):
        super().__init__()
        self.config = config
        self.bert_config = bert_config
        self.n_use_layer = 4
        self.dropout = nn.Dropout(config.dropout)
        self.dense1 = nn.Linear(
            self.bert_config.hidden_size * self.n_use_layer,
            self.bert_config.hidden_size * self.n_use_layer)
        self.dense2 = nn.Linear(
            self.bert_config.hidden_size * self.n_use_layer,
            self.bert_config.hidden_size * self.n_use_layer)
        self.classifier = nn.Linear(
            self.bert_config.hidden_size * self.n_use_layer,
            self.config.ocnli['num_classes'])

    def forward(self, embeddings):

        outputs = embeddings[2]
        pooled_output = torch.cat(
            [outputs[2][-1 * i][:, 0] for i in range(1, self.n_use_layer + 1)],
            dim=1)
        pooled_output = self.dense1(pooled_output)
        pooled_output = self.dense2(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class TextTransformer(nn.Module):
    """
    利用两层transformer encoder捕捉语义信息
    利用倒数第二层的向量与transformer捕捉的信息进行multi sample dropout
    每层dropout前都进行平均池化或者最大池化
    """

    def __init__(self, config, bert_config):
        self.config = config
        self.bert_config = bert_config
        self.transformer_layer = nn.TransformerEncoderLayer(
            self.config.hidden_size, self.config.attention_head)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=self.config.num_layers)
        self.low_dropout = nn.Dropout(self.config.low_dropout)
        self.high_dropout = nn.Dropout(self.config.high_dropout)
        self.classifier = nn.Linear(
            self.bert_config.hidden_size, self.config.ocnli['num_classes'])

    def forward(self, output):
        hidden_layers = output[2]
        sample_output = []
        # 取最后四层embedding
        for i in range(1, 5):
            sentence_encoding = self.transformer_encoder(
                self.low_dropout(hidden_layers[-i][:, 0, :]))
            sample_output.append(sentence_encoding)
        # 最后四层分别做分类，然后取均值
        out = torch.mean(
            torch.stack(
                [
                    self.classifier(self.high_dropout(x))
                    for x in sample_output
                ],
                dim=0,
            ),
            dim=0,
        )
        return out
