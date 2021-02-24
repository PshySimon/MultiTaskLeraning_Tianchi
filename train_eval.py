"""
@Author:    Pshy Simon
@Date:  2020/12/1 0001 下午 07:07
@Description:
    训练函数
"""
import os
import torch
import zipfile
from tqdm import tqdm
from transformers import AdamW
from sklearn.metrics import f1_score
from losses import DynamicWeightAveragingLoss, DynamicTaskPrioritizationLoss
from adversial import FGM, FreeLB, PGD
from optimizer import PCGrad
from typing import Tuple
from utils.Utils import OCEMOTION, OCNLI, TNEWS, TASK


def train(config, train_iter, dev_iter, test_iter, Model, label_weights=None):

    save_path = "saved_dict"
    checkpoint_path = os.path.join(save_path, 'checkpoint.pkl')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 模型准备
    best_f1 = 0.
    flag = False
    stop_steps = 0
    early_stop = 4000
    model = Model(config, label_weights)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
       'weight_decay': config.weight_decay},
      {'params': [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # 再次针对不同的层设置不同的学习率
    leraning_rate_decay = config.bert_learning_rate_decay
    last_bert_learning_rate = config.bert_learning_rate
    optimized_learning_rate_params = []
    for param_group in optimizer_grouped_parameters:
        p, w = param_group['params'], param_group['weight_decay']
        other_layers = []
        for name, param in p:
            if 'encoder' in name:
                optimized_learning_rate_params.append({'params':[param], 'lr': last_bert_learning_rate, 'weight_decay': w})
                last_bert_learning_rate = last_bert_learning_rate*leraning_rate_decay
            else:
                other_layers.append(param)
        optimized_learning_rate_params.append({'params':other_layers, 'lr': config.learning_rate, 'weight_decay': w})
    optimizer = AdamW(optimized_learning_rate_params, eps=config.adam_epsilon)

    # 如果检测到当前checkpoint，则读取checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.to(config.device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        tqdm.write("从断点处恢复训练，当前epch：{}".format(start_epoch))
    else:
        start_epoch = 0
        model.to(config.device)
    if config.use_pcgrad:
        optimizer = PCGrad.PCGrad(optimizer) 

    # ---------------------声明选用哪种loss优化策略以及对抗训练策略-------------------------
    if config.loss_strategy == "None":
        loss_strategy = None
    elif config.use_pcgrad:
        raise RuntimeError("使用的loss优化策略{}与pcgrad不兼容".format(config.loss_strategy))
    elif config.loss_strategy == "dwa":
        loss_strategy = DynamicWeightAveragingLoss.DynamicWeightAveragingLoss(model)
    elif config.loss_strategy == "dtp":
        loss_strategy = DynamicTaskPrioritizationLoss.DynamicTaskPrioritizationLoss(model)
    else:
        raise RuntimeError("没有该对loss优化练策略{}".format(config.loss_strategy))

    if config.adversial == "None":
        adversial = None
    elif config.adversial == "fgm":
        adversial = FGM.FGM(model)
    elif config.adversial == "pgd":
        adversial = PGD.PGD(model)
    elif config.adversial == "freelb":
        adversial = FreeLB.FreeLB(model)
    else:
        raise RuntimeError("没有该对抗训练策略{}".format(config.adversial))
    
    # ---------------------声明选用哪种loss优化策略以及对抗训练策略-------------------------

    for epoch in range(start_epoch, config.num_epochs):
        tqdm.write("------------------------EPOCH[{}/{}]------------------------".format(epoch+1, config.num_epochs))
        step = 0
        total_loss = 0.
        train_ocnli_pred_list = []
        train_ocnli_label_list = []
        train_tnews_pred_list = []
        train_tnews_label_list = []
        train_ocemotion_pred_list = []
        train_ocemotion_label_list = []

        with tqdm(total=len(train_iter[0])) as pbar:
            for ocemotion_batch, ocnli_batch, tnews_batch in zip(*train_iter):
                pbar.update(1 if not config.background else 0)
                model.train()
                step += 1
                ocemotion_batch = [x.to(config.device) for x in ocemotion_batch]
                ocnli_batch = [x.to(config.device) for x in ocnli_batch]
                tnews_batch = [x.to(config.device) for x in tnews_batch]
                ocemotion_out, ocemotion_loss = model(*ocemotion_batch, OCEMOTION)
                ocnli_out, ocnli_loss = model(*ocnli_batch, OCNLI)
                tnews_out, tnews_loss = model(*tnews_batch, TNEWS)
                ocemotion_label, ocnli_label, tnews_label = ocemotion_batch[-1], ocnli_batch[-1], tnews_batch[-1]

                # --------------loss--------------------
                params = {'model': model,
                          'ocemotion_batch': ocemotion_batch,
                          'ocnli_batch': ocnli_batch,
                          'tnews_batch': tnews_batch,
                          'step': step,
                          'ocemotion_loss': ocemotion_loss,
                          'ocnli_loss': ocnli_loss,
                          'tnews_loss': tnews_loss,
                          'ocemotion_out': ocemotion_out,
                          'ocnli_out': ocnli_out,
                          'tnews_out': tnews_out,
                          'ocemotion_label': ocemotion_label,
                          'ocnli_label': ocnli_label,
                          'tnews_label': tnews_label}
                
                if loss_strategy is not None:
                    loss = loss_strategy.get_loss(**params)
                else:
                    loss = model.compute_loss(ocemotion_loss, ocnli_loss, tnews_loss)
                # --------------loss--------------------

                if isinstance(loss, list):
                    total_loss += sum([x.item() for x in loss])
                    optimizer.pc_backward(loss)
                else:
                    total_loss += loss.item()
                    loss.backward()

                # --------------adversial-------------------------
                if adversial is not None:
                    adversial.train(**params)
                # --------------adversial-------------------------

                optimizer.step()
                model.zero_grad()

                for label, out, preds, labels in zip((ocemotion_label, ocnli_label, tnews_label),
                                                     (ocemotion_out, ocnli_out, tnews_out),
                                                     (train_ocemotion_pred_list, train_ocnli_pred_list, train_tnews_pred_list),
                                                     (train_ocemotion_label_list, train_ocnli_label_list, train_tnews_label_list)):
                    label = label.float().detach().cpu().numpy().tolist()
                    predict = torch.argmax(out, dim=-1).float().detach().cpu().numpy().tolist()
                    preds.extend(predict)
                    labels.extend(label)

                msg = "ITER:{}, TRAIN_LOSS:{:.3f}, TRAIN_F1:{:.2%}, emotion_train_f1:{:.2%}, nli_train_f1:{:.2%},"\
                    "tnews_train_f1:{:.2%}, DEV_LOSS:{:.3f},DEV_F1:{:.2%}, emotion_dev_f1:{:.2%},"\
                    "nli_dev_f1:{:.2%}, tnews_dev_f1:{:.2%}{}\n"
                if step % 500 == 0:
                    f1_train_emotion = f1_score(train_ocemotion_label_list, train_ocemotion_pred_list, average='macro')
                    f1_train_nli = f1_score(train_ocnli_label_list, train_ocnli_pred_list, average='macro')
                    f1_train_tnews = f1_score(train_tnews_label_list, train_tnews_pred_list, average='macro')
                    f1_score_train = (f1_train_emotion + f1_train_nli + f1_train_tnews) / 3
                    dev_loss, dev_f1, emotion_dev_f1, nli_dev_f1, tnews_dev_f1 = evaluate(config, dev_iter, model)

                    if dev_f1 > best_f1:
                        best_f1 = dev_f1
                        stop_steps = 0
                        improvement = True
                        torch.save({'state_dict': model.state_dict()}, os.path.join(save_path, "check_point_dev_loss.pkl"))
                    else:
                        improvement = False
                    tqdm.write(msg.format(step, total_loss / step, f1_score_train, f1_train_emotion, f1_train_nli,
                               f1_train_tnews, dev_loss, dev_f1, emotion_dev_f1, nli_dev_f1, tnews_dev_f1,
                               "*" if improvement else " "))
                stop_steps += 1
                if stop_steps > early_stop:
                    tqdm.write("more than {} steps not improved yet, early stopping".format(stop_steps))
                    flag = True
                    break
            # 每个epoch都将模型、优化器和epoch记录下来
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, checkpoint_path)
            if flag:
                break
    tqdm.write("----------------------------testing-------------------------------")
    checkpoint = torch.load('saved_dict/check_point_dev_loss.pkl')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(config.device)
    predictions = test(config, test_iter, model)
    tqdm.write("-----------------------------------------------------------")
    del model
    torch.cuda.empty_cache()
    os.remove(checkpoint_path)
    return predictions


def evaluate(config, dev_iter, model):
    all_pred_e, all_true_e, all_pred_n, all_true_n, all_pred_t, all_true_t = [], [], [], [], [], []
    total_loss = 0.

    def calc(dev, _task):
        with torch.no_grad():
            step = 0
            tmp_loss = 0.
            for batch in dev:
                model.eval()
                step += 1
                batch = [x.to(config.device) for x in batch]
                out, loss = model(*batch, _task)
                tmp_loss += loss.item()
                task_pred = torch.argmax(
                    out, dim=-1).float().detach().cpu().numpy().tolist()
                task_true = batch[-1].detach().cpu().numpy().tolist()
                if task == OCEMOTION:
                    all_pred_e.extend(task_pred)
                    all_true_e.extend(task_true)
                elif task == OCNLI:
                    all_pred_n.extend(task_pred)
                    all_true_n.extend(task_true)
                else:
                    all_pred_t.extend(task_pred)
                    all_true_t.extend(task_true)
        return tmp_loss / step

    # 三个任务分开计算loss和f1
    for task in range(len(dev_iter)):
        task_loss = calc(dev_iter[task], task)
        total_loss += task_loss

    f1_score_t = f1_score(all_true_t, all_pred_t, average='macro')
    f1_score_n = f1_score(all_true_n, all_pred_n, average='macro')
    f1_score_e = f1_score(all_true_e, all_pred_e, average='macro')
    return total_loss / 3, (f1_score_t + f1_score_e + f1_score_n) / 3, f1_score_e, f1_score_n, f1_score_t


def test(config, test_iter, model):
    all_pred = []
    for task, task_iter in enumerate(test_iter):
        pred = []
        for i, batch in enumerate(tqdm(task_iter)):
            model.eval()
            with torch.no_grad():
                batch = [x.to(config.device) for x in batch]
                input_ids, attention_masks, token_type_ids = batch
                out = model(input_ids, attention_masks,
                            token_type_ids, None, task)
                y_pred = torch.argmax(
                    out[0], dim=-1).detach().cpu().numpy().tolist()
                pred.extend(y_pred)
        all_pred.append(pred)
    return all_pred


def submit(all_predictions):
    if not os.path.exists("output"):
        os.mkdir("output")
    print("正在生成预测文件...")
    emotion_labels = ['sadness', 'happiness', 'like', 'anger', 'fear', 'surprise', 'disgust']
    nli_labels = ['0', '1', '2']
    tnews_labels = ['108', '104', '106', '112', '109', '103', '116', '101', '107', '100', '102', '110', '115', '113', '114']

    for task in range(len(TASK)):
        predict = all_predictions[task]
        if task == OCEMOTION:
            filename = "output/ocemotion_predict.json"
            labels = emotion_labels
        elif task == OCNLI:
            filename = "output/ocnli_predict.json"
            labels = nli_labels
        else:
            filename = "output/tnews_predict.json"
            labels = tnews_labels
        with open(filename, 'w', encoding='utf-8') as fin:
            for idx, pred in enumerate(predict):
                # pred是一个长度为标签数量的数组
                label = labels[pred]
                fin.write('{"id":%d, "label": "%s"}\n' % (idx, label))

    output = 'output/submit.zip'
    zip_files = ["output/ocemotion_predict.json", "output/ocnli_predict.json", "output/tnews_predict.json"]

    # 遍历files文件夹下的文件，压缩发送
    def compress_attaches(files, out_name):
        f = zipfile.ZipFile(out_name, 'w', zipfile.ZIP_DEFLATED)
        for file in files:
            f.write(file)
        f.close()

    compress_attaches(zip_files, output)
