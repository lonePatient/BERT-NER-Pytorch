#encoding:utf-8
import torch
import warnings
from pytorch_pretrained_bert.optimization import BertAdam
from pybert.train.metrics import AUCThresh
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.dataset import CreateDataset
from pybert.io.data_transformer import DataTransformer
from pybert.utils.logginger import init_logger
from pybert.utils.utils import seed_everything
from pybert.config.basic_config import configs as config
from pybert.callback.lrscheduler import BertLr
from pybert.model.nn.bert_fine import BertFine
from pybert.preprocessing.preprocessor import Preprocessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
warnings.filterwarnings("ignore")

# 主函数
def main():
    # **************************** 基础信息 ***********************
    logger = init_logger(log_name=config['arch'], log_dir=config['log_dir'])
    logger.info("seed is %d"%config['seed'])
    device = 'cuda:%d' % config['n_gpus'][0] if len(config['n_gpus']) else 'cpu'
    seed_everything(seed=config['seed'],device=device)
    logger.info('starting load data from disk')
    config['id_to_label'] = {v:k for k,v in config['label_to_id'].items()}
    # **************************** 数据生成 ***********************
    data_transformer = DataTransformer(logger      = logger,
                                       raw_data_path=config['raw_data_path'],
                                       label_to_id = config['label_to_id'],
                                       train_file  = config['train_file_path'],
                                       valid_file  = config['valid_file_path'],
                                       valid_size  = config['valid_size'],
                                       seed        = config['seed'],
                                       preprocess  = Preprocessor(),
                                       shuffle     = True,
                                       skip_header = True,
                                       stratify    = False)
    # 读取数据集以及数据划分
    data_transformer.read_data()
    # train
    train_dataset   = CreateDataset(data_path    = config['train_file_path'],
                                    vocab_path   = config['vocab_path'],
                                    max_seq_len  = config['max_seq_len'],
                                    seed         = config['seed'],
                                    example_type = 'train')
    # valid
    valid_dataset   = CreateDataset(data_path    = config['valid_file_path'],
                                    vocab_path   = config['vocab_path'],
                                    max_seq_len  = config['max_seq_len'],
                                    seed         = config['seed'],
                                    example_type = 'valid')
    #加载训练数据集
    train_loader = DataLoader(dataset     = train_dataset,
                              batch_size  = config['batch_size'],
                              num_workers = config['num_workers'],
                              shuffle     = True,
                              drop_last   = False,
                              pin_memory  = False)
    # 验证数据集
    valid_loader = DataLoader(dataset     = valid_dataset,
                              batch_size  = config['batch_size'],
                              num_workers = config['num_workers'],
                              shuffle     = False,
                              drop_last   = False,
                              pin_memory  = False)

    # **************************** 模型 ***********************
    logger.info("initializing model")
    model = BertFine.from_pretrained(config['bert_model_dir'],
                                     cache_dir=config['cache_dir'],
                                     num_classes = len(config['label_to_id']))

    # ************************** 优化器 *************************
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_steps = int(
        len(train_dataset.examples) / config['batch_size'] / config['gradient_accumulation_steps'] * config['epochs'])
    # t_total: total number of training steps for the learning rate schedule
    # warmup: portion of t_total for the warmup
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = config['learning_rate'],
                         warmup = config['warmup_proportion'],
                         t_total = num_train_steps)

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")
    # 模型保存
    model_checkpoint = ModelCheckpoint(checkpoint_dir   = config['checkpoint_dir'],
                                       mode             = config['mode'],
                                       monitor          = config['monitor'],
                                       save_best_only   = config['save_best_only'],
                                       best_model_name  = config['best_model_name'],
                                       epoch_model_name = config['epoch_model_name'],
                                       arch             = config['arch'],
                                       logger           = logger)
    # 监控训练过程
    train_monitor = TrainingMonitor(fig_dir  = config['figure_dir'],
                                    json_dir = config['log_dir'],
                                    arch     = config['arch'])
    # 学习率机制
    lr_scheduler = BertLr(optimizer = optimizer,
                          lr        = config['learning_rate'],
                          t_total   = num_train_steps,
                          warmup    = config['warmup_proportion'])

    # **************************** training model ***********************
    logger.info('training model....')
    trainer = Trainer(model            = model,
                      train_data       = train_loader,
                      val_data         = valid_loader,
                      optimizer        = optimizer,
                      epochs           = config['epochs'],
                      criterion        = BCEWithLogLoss(),
                      logger           = logger,
                      model_checkpoint = model_checkpoint,
                      training_monitor = train_monitor,
                      resume           = config['resume'],
                      lr_scheduler     = lr_scheduler,
                      n_gpu            = config['n_gpus'],
                      evaluate         = AUCThresh(thresh=0.5,sigmoid=True))
    # 查看模型结构
    trainer.summary()
    # 拟合模型
    trainer.train()
    # 释放显存
    if len(config['n_gpus']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
