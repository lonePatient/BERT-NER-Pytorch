#encoding:utf-8
from os import path
import multiprocessing

"""Note:
pytorch BERT 模型包含三个文件：模型、vocab.txt, bert_config.json, 有两种加载方式：
（1）在线下载。这种方式下，模型和vocab会通过url的方式下载，只需将bert_model设置为 "bert_model=bert-base-chinese"
     另外，还需要设置cache_dir路径，用来存储下载的文件。
（2）先下载好文件。下载好的文件是tensorflow的ckpt格式的，首先要利用convert_tf_checkpoint_to_pytorch转换成pytorch格式存储
     这种方式是通过本地文件夹直接加载的，要注意这时的文件命名方式。首先指定bert_model=存储模型的文件夹
     第二，将vocab.txt和bert_config.json放入该目录下，并在配置文件中指定VOCAB_FILE路径。当然vocab.txt可以不和模型放在一起，
     但是bert_config.json文件必须和模型文件在一起。具体可见源代码file_utils
"""

BASE_DIR = 'pybert'

configs = {
    'arch':'bert-multi-label',
    'raw_data_path': path.sep.join([BASE_DIR,'dataset/raw/source_BIO_2014_cropus.txt']),   # 总的数据，一般是将train和test何在一起构建语料库
    'raw_target_path': path.sep.join([BASE_DIR,'dataset/raw/target_BIO_2014_cropus.txt']), #　原始的标签数据
    'train_file_path': path.sep.join([BASE_DIR,'dataset/processed/train.tsv']),
    'valid_file_path': path.sep.join([BASE_DIR,'dataset/processed/valid.tsv']),

    'log_dir': path.sep.join([BASE_DIR, 'output/log']), # 模型运行日志
    'writer_dir': path.sep.join([BASE_DIR, 'output/TSboard']),# TSboard信息保存路径
    'figure_dir': path.sep.join([BASE_DIR, 'output/figure']), # 图形保存路径
    'checkpoint_dir': path.sep.join([BASE_DIR, 'output/checkpoints']),# 模型保存路径
    'cache_dir': path.sep.join([BASE_DIR,'model/']),

    'vocab_path': path.sep.join([BASE_DIR, 'model/pretrain/bert-base-uncased/vocab.txt']),
    'tf_checkpoint_path': path.sep.join([BASE_DIR, 'model/pretrain/bert-base-uncased/bert_model.ckpt']),
    'bert_config_file': path.sep.join([BASE_DIR, 'model/pretrain/bert-base-uncased/bert_config.json']),
    'pytorch_model_path': path.sep.join([BASE_DIR, 'model/pretrain/pytorch_pretrain/pytorch_model.bin']),
    'bert_model_dir': path.sep.join([BASE_DIR, 'model/pretrain/pytorch_pretrain']),

    'valid_size': 0.1, # valid数据集大小
    'max_seq_len': 512,  # word文本平均长度,按照覆盖95%样本的标准，取截断长度:np.percentile(list,95.0)

    'batch_size': 16,   # how many samples to process at once
    'epochs': 5,       # number of epochs to train
    'start_epoch': 1,
    'warmup_proportion': 0.1, # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
    'gradient_accumulation_steps':1, # Number of updates steps to accumulate before performing a backward/update pass.
    'learning_rate': 2e-5,
    'n_gpus': [1,0], # GPU个数,如果只写一个数字，则表示gpu标号从0开始，并且默认使用gpu:0作为controller,
                     # 如果以列表形式表示，即[1,3,5],则我们默认list[0]作为controller

    'num_workers': multiprocessing.cpu_count(), # 线程个数
    'resume':False,
    'seed': 2018,
    'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
    'mode': 'min',    # one of {min, max}
    'monitor': 'val_loss',  # 计算指标
    'early_patience': 10,   # early_stopping
    'save_best_only': True, # 是否保存最好模型
    'best_model_name': '{arch}-best2.pth', #保存文件
    'epoch_model_name': '{arch}-{epoch}-{val_loss}.pth', #以epoch频率保存模型
    'save_checkpoint_freq': 10, #保存模型频率，当save_best_only为False时候，指定才有作用

    'label_to_id': {  # 标签映射
        "B_PER": 1,  # 人名
        "I_PER": 2,
        "B_LOC": 3,  # 地点
        "I_LOC": 4,
        "B_ORG": 5,  # 机构
        "I_ORG": 6,
        "B_T": 7,  # 时间
        "I_T": 8,
        "O": 9,  # 其他
        "BOS": 10,  # 起始符
        "EOS": 11  # 结束符
    },
}
