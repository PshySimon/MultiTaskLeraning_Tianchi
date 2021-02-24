"""
@Author:    Pshy Simon
@Date:  2020/12/10 0010 上午 09:38
@Description:
    数据迭代器：按照数据比例缩放每个任务的batch_size
"""
import os
import json
import torch
import random
import dataclasses
import pandas as pd
from tqdm import tqdm
from math import floor
from typing import Optional, List
from dataclasses import dataclass
from .DataSpliter import DataSpliter
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from .Utils import TASK, OCEMOTION, OCNLI, TNEWS, TRAIN, DEV, TEST
random.seed(2020)


@dataclass
class InputExamples:
    """
    通用的抽取任务数据的模板
    """
    ids: int
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class DataGenerator:
    def __init__(self, config):
        # 三个任务的样本，每个样本包括三个数据集：train,dev和test
        self.task_data = []
        self.task_features = []
        self.train_data_num = 0
        self.train_emotion_num = 0
        self.train_nli_num = 0
        self.train_tnews_num = 0
        self.dev_emotion_num = 0
        self.dev_nli_num = 0
        self.dev_tnews_num = 0
        self.test_emotion_num = 0
        self.test_nli_num = 0
        self.test_tnews_num = 0
        self.config = config
        self.batch_sizes, self.label_weights = DataSpliter.split_file(config)
        self.emotion_labels = {'sadness': 0, 'happiness': 1, 'like': 2, 'anger': 3, 'fear': 4, 'surprise': 5, 'disgust': 6}
        self.tnews_labels = {108: 0, 104: 1, 106: 2, 112: 3, 109: 4, 103: 5, 116: 6, 101: 7, 107: 8,
                             100: 9, 102: 10, 110: 11, 115: 12, 113: 13, 114: 14}
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)

    def load_and_cache_examples(self):
        """
        加载数据集到内存中
        :return
        """
        def _fetch_examples(data: pd.DataFrame, task: int) -> List[InputExamples]:
            examples = []
            if task == OCEMOTION:
                label_map = self.emotion_labels
            elif task == OCNLI:
                label_map = None
            else:
                label_map = self.tnews_labels
            for i, row in data.iterrows():
                try:
                    text_b = row['text_b']
                except KeyError:
                    text_b = None
                try:
                    label = row['labels']
                    label = label_map[label] if label_map is not None else label
                except KeyError:
                    label = None
                res = {'ids': row['id'], 'text_a': row['text_a'], 'text_b': text_b, 'label': label}
                examples.append(InputExamples(**{x[0]: x[1] for x in res.items() if x[1] is not None}))
            return examples

        read_cols = 1000 if self.config.debug else None
        # 分别对三个任务进行样本的读取和缓存
        for t, tname in enumerate(TASK):
            task = []
            # 分别对train,dev和test进行读取
            for p in ['train.csv', 'dev.csv', 'test.csv']:
                path = os.path.join("./data/"+tname, p)
                data = pd.read_csv(path, sep='\t', quoting=3, nrows=read_cols)
                examples = _fetch_examples(data, t)
                task.append(examples)
            self.task_data.append(task)
        self.train_data_num = sum([len(x[0]) for x in self.task_data])

        self.train_emotion_num = len(self.task_data[OCEMOTION][TRAIN])
        self.train_nli_num = len(self.task_data[OCNLI][TRAIN])
        self.train_tnews_num = len(self.task_data[TNEWS][TRAIN])

        self.dev_emotion_num = len(self.task_data[OCEMOTION][DEV])
        self.dev_nli_num = len(self.task_data[OCNLI][DEV])
        self.dev_tnews_num = len(self.task_data[TNEWS][DEV])

        self.test_emotion_num = len(self.task_data[OCEMOTION][TEST])
        self.test_nli_num = len(self.task_data[OCNLI][TEST])
        self.test_tnews_num = len(self.task_data[TNEWS][TEST])

    # 对句子对进行截断的技巧
    def _trim_seq_pair(self, question_tokens, answer_tokens, max_sequence_length, q_max_len, a_max_len):
        q_len = len(question_tokens)
        a_len = len(answer_tokens)
        if q_len + a_len + 3 > max_sequence_length:
            if a_max_len <= a_len and q_max_len <= q_len:
                q_new_len_head = floor((q_max_len - q_max_len / 2))
                question_tokens = question_tokens[:q_new_len_head] + question_tokens[q_new_len_head - q_max_len:]
                a_new_len_head = floor((a_max_len - a_max_len / 2))
                answer_tokens = answer_tokens[:a_new_len_head] + answer_tokens[a_new_len_head - a_max_len:]
            elif q_len <= a_len and q_len < q_max_len:
                a_max_len = a_max_len + (q_max_len - q_len - 1)
                a_new_len_head = floor((a_max_len - a_max_len / 2))
                answer_tokens = answer_tokens[:a_new_len_head] + answer_tokens[a_new_len_head - a_max_len:]
            elif a_len < q_len:
                assert a_len <= a_max_len
                q_max_len = q_max_len + (a_max_len - a_len - 1)
                q_new_len_head = floor((q_max_len - q_max_len / 2))
                question_tokens = question_tokens[:q_new_len_head] + question_tokens[q_new_len_head - q_max_len:]
            else:
                self._truncate_seq_pair(question_tokens, answer_tokens, max_sequence_length - 3)
        return question_tokens, answer_tokens

        # 谷歌的对文本进行截断的代码
    @staticmethod
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_examples_into_input_ids_for_classification(self, text, max_length):
        """
        文本分类用的数据转化方法，用于文本分类和情感分析
        :param text: 字符文本
        :return: input_ids, attention_mask, token_type_ids
        """
        # 太长的句子直接截断
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]
        text_len = len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens + ['[PAD]'] * (max_length - text_len))
        attention_mask = [1] * text_len + [0] * (max_length - text_len)
        token_type_ids = [0] * max_length

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        return input_ids, attention_mask, token_type_ids

    def _convert_examples_into_input_ids_for_matching(self, text_a, text_b, max_length, max_length_a, max_length_b):
        """
        对句子对做处理
        :param text_a: 第一个句子
        :param text_b: 第二个句子
        :param max_length: 总体长度
        :param max_length_a: 第一个句子长度
        :param max_length_b: 第二个句子长度
        :return: input_ids, attention_mask, token_type_ids
        """
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        tokens_a, tokens_b = self._trim_seq_pair(tokens_a, tokens_b, max_length, max_length_a, max_length_b)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        text_len = len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens + ["[PAD]"] * (max_length - text_len))
        attention_mask = [1] * text_len + [0] * (max_length - text_len)
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) + [0] * (max_length - text_len)
        return input_ids, attention_mask, token_type_ids

    def build_features(self):
        """
        将三个任务的样本转化成tokens
        :return:
        """
        bs_OCEMOTION, bs_OCNLI, bs_TNEWS = self.batch_sizes
        # 加载三个任务的数据集
        for task in range(len(self.task_data)):
            # 存储三个任务的train, dev和test的data loader
            task_feature = []
            if task != OCNLI:
                max_length = self.config.ocemotion['max_length'] if task == OCEMOTION else\
                    self.config.tnews['max_length']
                # 遍历训练集、验证集和测试集
                for idx, ds in enumerate(self.task_data[task]):
                    all_input_ids, all_attention_mask, all_token_type_ids, all_labels = [], [], [], []
                    iterator = tqdm(ds) if not self.config.background else ds
                    for example in iterator:
                        text_a = example.text_a
                        label = example.label
                        input_id, att_mask, token_type_id = self._convert_examples_into_input_ids_for_classification(
                            text_a, max_length)
                        if label is not None:
                            all_labels.append(label)
                        all_input_ids.append(input_id)
                        all_attention_mask.append(att_mask)
                        all_token_type_ids.append(token_type_id)
                    all_input_ids1 = torch.tensor(all_input_ids, dtype=torch.long)
                    all_attention_mask1 = torch.tensor(all_attention_mask, dtype=torch.long)
                    all_token_type_ids1 = torch.tensor(all_token_type_ids, dtype=torch.long)
                    all_labels = torch.tensor(all_labels, dtype=torch.long) if len(all_labels) > 1 else None
                    dataset = TensorDataset(*[x for x in (all_input_ids1, all_attention_mask1,
                                                          all_token_type_ids1, all_labels) if x is not None])
                    data_loader = DataLoader(dataset, batch_size=bs_OCEMOTION if task == OCEMOTION else bs_TNEWS,
                                             shuffle=True if idx != TEST else False, num_workers=4)
                    task_feature.append(data_loader)
            else:
                max_length = self.config.ocnli['max_length']
                max_length_a = self.config.ocnli['max_length_a']
                max_length_b = self.config.ocnli['max_length_b']
                for idx, ds in enumerate(self.task_data[task]):
                    all_input_ids, all_attention_mask, all_token_type_ids, all_labels = [], [], [], []
                    iterator = tqdm(ds) if not self.config.background else ds
                    for example in iterator:
                        text_a = example.text_a
                        text_b = example.text_b
                        label = example.label
                        input_id, att_mask, token_type_id = self._convert_examples_into_input_ids_for_matching(
                            text_a, text_b, max_length, max_length_a, max_length_b
                        )
                        if label is not None:
                            all_labels.append(label)
                        all_input_ids.append(input_id)
                        all_attention_mask.append(att_mask)
                        all_token_type_ids.append(token_type_id)
                    all_input_ids1 = torch.tensor(all_input_ids, dtype=torch.long)
                    all_attention_mask1 = torch.tensor(all_attention_mask, dtype=torch.long)
                    all_token_type_ids1 = torch.tensor(all_token_type_ids, dtype=torch.long)
                    all_labels = torch.tensor(all_labels, dtype=torch.long) if len(all_labels) > 1 else None

                    dataset = TensorDataset(*[x for x in (all_input_ids1, all_attention_mask1,
                                                          all_token_type_ids1, all_labels) if x is not None])
                    data_loader = DataLoader(dataset, batch_size=bs_OCNLI, shuffle=True if idx != TEST else False, num_workers=4)
                    task_feature.append(data_loader)
            self.task_features.append(task_feature)

    def get_iter(self, ds):
        # 一次产生一个批量的数据
        return [task[ds] for task in self.task_features]
