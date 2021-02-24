"""
@Author:    Pshy Simon
@Date:  2020/12/10 0010 上午 10:31
@Description:
   将数据集划分为训练集和验证集
"""
import random
import os
import pandas as pd
import shutil
from typing import List
from tqdm import tqdm
from math import log
from .Utils import TASK

random.seed(2020)


class DataSpliter:
    # 根据batch_size寻找一种策略，使得三个训练集批数是一致的，且要保证验证集至少占数据的十分之一
    @staticmethod
    def fetch_dev_nums(train_nums: List[int], batch_size, min_percentage=0.1):
        x, y, z = train_nums
        total = x + y + z
        # 各个数据集占总数据集的比例
        p_x, p_y, p_z = x / total, y / total, z / total
        # 各个数据集可取的batch_size
        b_x, b_y, b_z = round(batch_size * p_x), round(batch_size * p_y), round(batch_size * p_z)
        # 各个数据集跑的最少批次
        (min_train_num, min_batch_num, min_batch_size) = min(
            [(x, x // b_x, b_x), (y, y // b_y, b_y), (z, z // b_z, b_z)], key=lambda _: _[1])
        # 最少的批次的数据集的验证集数量至少得占min_percentage
        min_dev_batch_num = int(min_train_num * min_percentage) // min_batch_size
        all_train_batch_num = min_batch_num - min_dev_batch_num
        return (
                   x - all_train_batch_num * b_x,
                   y - all_train_batch_num * b_y,
                   z - all_train_batch_num * b_z
               ), (b_x, b_y, b_z)

    @staticmethod
    def split_handler(task: str, columns: List[str], dev_num: int):
        path = "./data/" + task
        data = pd.read_csv(os.path.join(path, "dataset.csv"), header=None, sep='\t', quoting=3)
        data.columns = columns
        test = pd.read_csv(os.path.join(path, "test.csv"), header=None, sep='\t', quoting=3)
        test.columns = columns[:-1]
        # 统计标签分布
        labels = data.labels.unique()
        label_counts = data.labels.value_counts()
        sample_nums = {}
        samples = []
        # 遍历不同标签的组
        for label in labels:
            sample_nums[label] = int(label_counts[label] * (dev_num / len(data)))
        # 对少的补齐
        total_sample_num = sum([x[1] for x in sample_nums.items()])
        if total_sample_num != dev_num:
            random_label = random.randint(0, len(labels) - 1)
            cur_label_num = sample_nums[labels[random_label]]
            left_over = dev_num - total_sample_num
            sample_nums[labels[random_label]] = cur_label_num + left_over
        # 对不同的标签的层进行分层采样
        for examples in data.groupby(['labels']):
            label_, data_ = examples
            indices = [i for i in range(len(data_))]
            sample_indices = random.sample(indices, sample_nums[label_])
            left_over = list(set(indices) - set(sample_indices))
            assert len(left_over) + len(sample_indices) == len(indices)
            samples.append([data_.iloc[left_over], data_.iloc[sample_indices]])
        # 将分层采样的数据合并起来
        train = pd.concat([x[0] for x in samples])
        dev = pd.concat([x[1] for x in samples])
        train_path = os.path.join(path, "train.csv")
        dev_path = os.path.join(path, "dev.csv")
        test_path = os.path.join(path, "test.csv")
        train.to_csv(train_path, sep='\t', index=False)
        dev.to_csv(dev_path, sep='\t', index=False)
        test.to_csv(test_path, sep='\t', index=False)
        tqdm.write("{}的验证集和训练集，分别保存在：{}和{}".format(task, train_path, dev_path))

    @staticmethod
    def split_file(config):
        # 遍历data文件夹下的csv文件
        data_path = "./data/"
        tmp = os.listdir(data_path)
        files = [x for x in tmp if not os.path.isdir(os.path.join(data_path, x)) and x.endswith(".csv")]

        headers = {
            "OCEMOTION": ['id', 'text_a', 'labels'],
            "OCNLI": ['id', 'text_a', 'text_b', 'labels'],
            "TNEWS": ['id', 'text_a', 'labels']
        }

        for x in files:
            for t in TASK:
                if t in x:
                    file_path = os.path.join(data_path, t)
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    target_type = "dataset.csv" if "_train" in x else "test.csv"
                    source_path = os.path.join(data_path, x)
                    target_path = os.path.join(file_path, target_type)
                    shutil.copy(source_path, target_path)
                    tqdm.write("复制文件：{}到：{}".format(source_path, target_path))
        task_train_nums = []
        label_weights = []
        for t in TASK:
            tmp = pd.read_csv(data_path + t + "/dataset.csv", sep='\t', header=None, quoting=3)
            tmp.columns = headers[t]
            # 统计各个标签的数量
            counts = tmp.labels.value_counts()
            total_counts = sum([x for _, x in counts.items()])
            counts = [log(total_counts / x) for _, x in counts.items()]
            label_weights.append(counts)
            task_train_nums.append(len(tmp))
        dev_nums, batch_sizes = DataSpliter.fetch_dev_nums(task_train_nums, batch_size=config.batch_size)
        for t, n in zip(TASK, dev_nums):
            DataSpliter.split_handler(t, headers[t], n)
        return batch_sizes, label_weights
