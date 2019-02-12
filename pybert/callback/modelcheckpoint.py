#encoding:utf-8
import os
import numpy as np
import torch
from ..utils.utils import ensure_dir

class ModelCheckpoint(object):

    def __init__(self, checkpoint_dir,
                 monitor,
                 logger,
                 arch,
                 save_best_only = True,
                 best_model_name = None,
                 epoch_model_name = None,
                 mode='min',
                 epoch_freq=1,
                 best = None):

        self.monitor = monitor
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.epoch_freq = epoch_freq
        self.arch = arch
        self.logger = logger
        self.best_model_name = best_model_name
        self.epoch_model_name = epoch_model_name
        self.use = 'on_epoch_end'

        # 计算模式
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        # 这里主要重新加载模型时候
        #对best重新赋值
        if best:
            self.best = best
        ensure_dir(self.checkpoint_dir.format(arch = self.arch))

    def step(self, state,current):
        # 是否保存最好模型
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.logger.info('\nEpoch %d: %s improved from %0.5f to %0.5f'% (state['epoch'], self.monitor, self.best,current))
                self.best = current
                state['best'] = self.best
                best_path = os.path.join(self.checkpoint_dir.format(arch = self.arch), self.best_model_name.format(arch = self.arch))
                torch.save(state, best_path)
        # 每隔几个epoch保存下模型
        else:
            filename = os.path.join(self.checkpoint_dir.format(arch = self.arch), self.epoch_model_name.format(arch=self.arch,
                                                                                epoch=state['epoch'],
                                                                                val_loss=state[self.monitor]
                                                                                )
                                    )
            if state['epoch'] % self.epoch_freq == 0:
                self.logger.info("\nEpoch %d: save model to disk."%(state['epoch']))
                torch.save(state, filename)
