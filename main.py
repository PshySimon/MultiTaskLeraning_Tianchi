"""
获取参数配置
"""
import json
import argparse
from train_eval import train, submit
from utils.Utils import TRAIN, DEV, TEST
from utils.DataGenerator import DataGenerator
from models.MultiTaskLearningModel import Model


parser = argparse.ArgumentParser()
# ----------------------------------------------------------通用参数部分------------------------------------------------------------------
parser.add_argument("-debug", type=bool, default=False, help="是否打开调试模式")
parser.add_argument("-background", type=bool, default=False, help="是否在后台运行，如果在后台运行就关闭tqdm输出")
parser.add_argument("-bert_path", type=str, default="Ernie", help="预训练语言模型的目录，默认为Ernie")
parser.add_argument("-tnews", type=dict, default={"max_length": 60, "num_classes": 15}, help="新闻分类的参数，包括新闻文本的最大长度、标签数目")
parser.add_argument("-ocemotion", type=dict, default={"max_length": 170, "num_classes": 7}, help="情感分析的参数，包括最大文本长度、标签数目")
parser.add_argument("-ocnli", type=dict, default={"max_length": 100, "max_length_a": 50, "max_length_b": 30, "num_classes": 3},
                    help="自然语言推理的参数，包括最大token长度，两个文本的最大长度和标签数目")
parser.add_argument("-device", type=str, default="cuda", help="训练的设备，有两个选项：cpu, cuda")
parser.add_argument("-batch_size", type=int, default=72, help="三个任务的batch_size总和，三个任务的batch_size是按照各自的样本比例来计算的")
parser.add_argument("-learning_rate", type=float, default=1e-4, help="分类器的学习率，默认为1e-4")
parser.add_argument("-bert_learning_rate", type=float, default=5e-5, help="学习率，默认为2e-5")
parser.add_argument("-bert_learning_rate_decay", type=float, default=0.99, help="bert分层学习率衰减系数，来源于邱锡鹏老师的How to Fine Tune Bert For Text Classfication?中对bert逐层设置递减的学习率")
parser.add_argument("-num_epochs", type=int, default=10, help="训练的次数，一般4到5个epoch就收敛了")
parser.add_argument("-weight_decay", type=float, default=1e-2, help="权重衰减")
parser.add_argument("-adam_epsilon", type=float, default=1e-8, help="adam的epsilon参数")
parser.add_argument("-warmup_rate", type=float, default=5e-2, help="warmup的比率")
parser.add_argument("-dropout", type=float, default=0.2, help="dropout正则化，在0到1之间")
parser.add_argument("-label_weight", type=bool, default=False, help="是否使用带权交叉熵损失函数")
parser.add_argument("-loss_strategy", type=str, default="None", help="是否对loss进行优化，有四个可选优化选项：dwa,dtp,uw和None")
parser.add_argument("-adversial", type=str, default="None", help="是否使用对抗验证，有四个可选优化选项：fgm,pgd,freelb和None")
parser.add_argument("-checkpoint", type=str, default="checkpoint", help="checkpoint的保存路径")
parser.add_argument("-stop_epochs", type=int, default=3, help="早停法，跑多少epoch之后没提升就停下来")
parser.add_argument("-use_label_weights", type=bool, default=False, help="是否在计算loss时使用label_weights")
parser.add_argument("-loss_weight_strategy", type=bool, default='uw', help="是否对loss量级进行优化")
parser.add_argument("-loss_function", type=str, default='cross_entropy', help="使用的损失函数，可选的有：cross_entropy, focal_loss, dice_loss")
parser.add_argument("-use_pcgrad", type=bool, default=False, help="是否使用PCGrad")
# ---------------------------------------------------------特定任务参数部分---------------------------------------------------------------
# 1.各个loss优化策略的参数
# 1.1 dwa
parser.add_argument("-dwa_T", type=int, default=2, help="动态权重平均算法，T取值为1时近似于softmax")
# 1.2 dtp
parser.add_argument("-dtp_y", type=float, default=0.5, help="动态任务优先级算法，y值默认取0.5")
# 1.3 uw
parser.add_argument("-uw_init", type=float, default=1.0, help="不确定性加权，初始化三个权重均为1.0")

# 2.各个对抗验证的参数
# 2.1 fgm
parser.add_argument("-fgm_epsilon", type=float, default=1.0, help="fgm的epsilon参数")
parser.add_argument("-fgm_embedding_name", type=str, default="word_embeddings", help="预训练语言模型的嵌入层参数名称")
# 2.2 pgd
parser.add_argument("-pgd_epsilon", type=float, default=1.0, help="pgd的epsilon参数")
parser.add_argument("-pgd_alpha", type=float, default=0.3, help="pgd的alpha参数")
parser.add_argument("-pgd_embedding_name", type=str, default="word_embeddings", help="预训练语言模型的嵌入层参数名称")
parser.add_argument("-pgd_k", type=int, default=3, help="pgd走的步数")
# 2.3 freelb
parser.add_argument("-freelb_ocemotion", type=list, default=[1.0, 'l2', 3, 1e-1, 6e-1],
                    help="ocemotion任务的freelb参数，传入一个列表，其参数依次为：adv_init_mag,norm_type,adv_steps,adv_lr,adv_max_norm")
parser.add_argument("-freelb_ocnli", type=list, default=[8e-2, 'l2', 3, 4e-2, 0],
                    help="ocnli任务的freelb参数，传入一个列表，其参数依次为：adv_init_mag,norm_type,adv_steps,adv_lr,adv_max_norm")
parser.add_argument("-freelb_tnews", type=list, default=[1.0, 'l2', 3, 1e-1, 6e-1],
                    help="tnews任务的freelb参数，传入一个列表，其参数依次为：adv_init_mag,norm_type,adv_steps,adv_lr,adv_max_norm")

# 3.各个模型的参数
# 3.0 共用部分参数
# 3.0.1 Attention+MLP
parser.add_argument("-attmlp_head", "-model_attention_mlp_attention_head", type=int, default=16, help="Attention+MLP模型的attention head数量")
# 3.0.2 MultiSample Dropout
parser.add_argument("-low_dropout", "-model_multi_sample_dropout_low", type=float, default=0.2, help="MultiSample Dropout low dropout")
parser.add_argument("-high_dropout", "-model_multi_sample_dropout_high", type=float, default=0.5, help="MultiSample Dropout high dropout")
# 3.1 ocemotion
# 3.1.1 Bert+RNN Attention
parser.add_argument("-rnn_num_layers", "-model_rnn_attention_num_layers", type=int, default=2, help="RNN的层数")
parser.add_argument("-rnn_hidden_size", "-model_rnn_attention_hidden_size", type=int, default=256, help="RNN Attention模型RNN隐藏层数量")
parser.add_argument("-rnn_hidden_size_dense", "-model_rnn_attention_hidden_size_dense", type=int, default=256,
                    help="RNN Attention模型dense层隐藏层数量")

# 3.2 ocnli


# 3.3 tnews
# 3.3.1 Text-CNN
parser.add_argument("-cnn_num_filters", "-model_text_cnn_num_filters", type=int, default=128, help="Text CNN中的filter数量")
parser.add_argument("-cnn_filter_sizes", "-model_text_cnn_filter_sizes", type=list, default=[2, 3, 4, 5], help="Text CNN中的filter尺寸")
# 3.3.2 Text-RCNN
parser.add_argument("-rcnn_num_layers", type=int, default=2, help="TEXT RCNN的RNN层数")
parser.add_argument("-rcnn_hidden_size", type=int, default=256, help="TEXT RCNN的隐藏层数量")

# ---------------------------------------------------------------分割线--------------------------------------------------------------------
config = parser.parse_args()

if __name__ == "__main__":
    # 参数配置打印
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(config))
    # 生成数据
    data = DataGenerator(config)
    # 加载和缓存样本
    data.load_and_cache_examples()
    # 存到dataloader中
    data.build_features()
    # 拿取训练集、验证集和测试集
    train_iter, dev_iter, test_iter = data.get_iter(TRAIN), data.get_iter(DEV), data.get_iter(TEST)
    all_predictions = train(config, train_iter, dev_iter, test_iter, Model, data.label_weights)
    submit(all_predictions)
