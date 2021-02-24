"""
@Author:    Pshy Simon
@Date:  2020/12/15 0015 下午 05:28
@Description:
   情感分析采取的模型：
        LSTM + Attention: LSTM加上注意力机制
        last_several_layers + pooled_output：最后基层加池化层做拼接特征
        pooled_output + BiLSTM + Avg Pool + Max Pool：池化层+双向LSTM+平均池化+最大池化的拼接
"""
from abc import ABC

import torch
import torch.nn as nn
from typing import Tuple


# attention + mlp
class AttentionMLP(nn.Module, ABC):
    """
    使用Attention + MLP做分类任务
    """

    def __init__(self, config, bert_config):
        super().__init__()
        self.config = config
        self.bert_config = bert_config
        self.atten_layer = nn.Linear(
            self.bert_config.hidden_size, self.config.attention_head)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.config.dropout)
        # 三个任务接不同的dense层
        self.dense_emotion = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.bert_config.hidden_size,
                      config.attention_head * config.ocemotion['num_classes']),
            nn.ReLU(inplace=True),
        )

    def forward(self, embedding):
        cls_emb = embedding[0][:, 0, :].squeeze(1)
        # attention_score = [batch_size, attention_heads]
        attention_score = self.atten_layer(cls_emb)
        # attention_score = [batch_size, 1, attention_heads]
        attention_score = self.dropout(
            self.softmax_d1(attention_score).unsqueeze(1))
        out = self.dense_emotion(cls_emb).contiguous().view(-1, self.config.attention_head,
                                                            self.config.ocemotion['num_classes'])
        out = torch.matmul(attention_score, out).squeeze(1)
        return out


class MultiSampleDropout(nn.Module, ABC):
    """
        对多有的cls_output做加权平均，然后做MultiSampleDropout
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
                                    self.config.ocemotion['num_classes'])

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


# BertRNN_Att
class RNNAttention(nn.Module, ABC):

    def __init__(self, config, bert_config):
        super().__init__()
        self.config = config
        self.bert_config = bert_config
        self.n_use_layer = self.config.rnn_num_layers
        self.rnn = nn.LSTM(self.bert_config.hidden_size*self.config.rnn_num_layers, 256, bidirectional=True,
                           num_layers=config.rnn_num_layers, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.w = nn.Parameter(torch.randn(2 * 256))
        self.fc1 = nn.Linear(256 * 2, 512)
        self.fc2 = nn.Linear(512, self.config.ocemotion['num_classes'])

    def forward(self, embeddings):
        if isinstance(embeddings, Tuple):
            embedded = torch.cat([embeddings[2][-1 * i] for i in range(1, self.n_use_layer + 1)],dim=-1)
        else:
            embedded = embeddings
        out, _ = self.rnn(embedded)
        # out = [batch_size, seq_len, hidden_size * num_directions]
        score = torch.matmul(out, self.w)
        att = torch.softmax(score, dim=1).unsqueeze(-1)
        # att = [batch_size, seq_len, 1]
        out = out * att
        # out = [batch_size, seq_len, hidden_size * 2]
        out = torch.sum(out, 1)
        # out = [batch_size, hidden_size * 2]
        out = torch.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
