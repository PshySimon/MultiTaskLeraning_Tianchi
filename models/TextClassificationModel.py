"""
@Author:    Pshy Simon
@Date:  2020/12/15 0015 下午 04:55
@Description:
    文本分类模型，主要考虑以下几种模型：
        attention + mlp：三个任务都一样用
        text_cnn
        text_rcnn
        text_rnn_attention
        text_avg_pool
        text_mean_pool
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
        self.atten_layer = nn.Linear(self.bert_config.hidden_size, self.config.attention_head)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.config.dropout)
        # 三个任务接不同的dense层
        self.dense_tnews = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.bert_config.hidden_size, config.attention_head * config.tnews['num_classes']),
            nn.ReLU(inplace=True),
        )

    def forward(self, embedding):
        cls_emb = embedding[0][:, 0, :].squeeze(1)
        # attention_score = [batch_size, attention_heads]
        attention_score = self.atten_layer(cls_emb)
        # attention_score = [batch_size, 1, attention_heads]
        attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
        # out = [batch_size, attention_heads, num_classes]
        out = self.dense_tnews(cls_emb).contiguous().view(-1, self.config.attention_head,
                                                          self.config.tnews['num_classes'])
        # out = [batch_size, num_classes]
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
                                    self.config.tnews['num_classes'])

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


# Bert-TextCNN
class TextCNN(nn.Module, ABC):
    def __init__(self, config, bert_config):
        super().__init__()
        self.config = config
        self.bert_config = bert_config
        self.convs = nn.ModuleList(
            [nn.Conv1d(self.bert_config.hidden_size, config.cnn_num_filters, x) for x in config.cnn_filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.cnn_num_filters * len(config.cnn_filter_sizes), self.config.tnews['num_classes'])
        self.relu = nn.ReLU()

    def pool(self, out, conv):
        out = self.relu(conv(out))
        max_pool = nn.MaxPool1d(out.shape[-1])
        out = max_pool(out)
        out = out.squeeze(2)
        return out

    def forward(self, embeddings):
        if isinstance(embeddings, Tuple):
            embedded = embeddings[0]
        else:
            embedded = embeddings
        # embedding = [batch_size, seq_len, emb_dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch_size, seq_len, emb_dim]
        output = [self.pool(embedded, conv) for conv in self.convs]
        # output = num_filter_sizes * [batch_size, num_filters]
        out = torch.cat(output, dim=1)
        # out = [batch_size, num_filter_sizes * num_filters]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# BertRCNN
class TextRCNN(nn.Module):
    
    def __init__(self, config, bert_config):
        super().__init__()
        self.config = config
        self.bert_config = bert_config
        self.n_use_layer = self.config.rcnn_num_layers
        self.rnn = nn.LSTM(self.bert_config.hidden_size*self.config.rcnn_num_layers, self.config.rcnn_hidden_size,
                           batch_first=True, bidirectional=True, num_layers=self.config.rcnn_num_layers)
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.rcnn_hidden_size*2, self.config.tnews['num_classes'])
        self.w = nn.Parameter(torch.randn(2*self.config.rcnn_hidden_size + self.bert_config.hidden_size*self.config.rcnn_num_layers,
                                          2 * self.config.rcnn_hidden_size))
        
    def forward(self, embeddings):
        if isinstance(embeddings, Tuple):
            embedded = torch.cat([embeddings[2][-1 * i] for i in range(1, self.n_use_layer + 1)],dim=-1)
        else:
            if self.n_use_layer == 1:
                embedded = embeddings
            else:
                raise RuntimeError("层数不匹配，无法使用freelb")
        # embedded = [batch_size, seq_len, 768*n_use_layer]
        out,_ = self.rnn(embedded)
        # 将输出和嵌入层连接起来
        # out = [batch_size, seq_len, 768*n_use_layer + rcnn_hidden_size*2]
        out = torch.cat((out, embedded), dim=2)
        # out = [batch_size, seq_len, hidden_size * 2 + emb_dim
        out = torch.tanh(torch.matmul(out, self.w))
        # out = [batch_size, seq_len, hidden_size * 2
        out = out.permute(0,2,1)
        # out = [batch_size, hidden_size * 2, seq_len]
        out = nn.functional.max_pool1d(out, out.shape[-1]).squeeze(2)
        out = self.fc(out)
        return out
