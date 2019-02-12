#encoding:utf-8
import os
import time
import numpy as np
import torch
from ..callback.progressbar import ProgressBar
from ..utils.utils import AverageMeter
from .train_utils import restore_checkpoint,model_device

# 训练包装器
class Trainer(object):
    def __init__(self,model,
                 train_data,
                 val_data,
                 optimizer,
                 epochs,
                 logger,
                 criterion,
                 evaluate,
                 lr_scheduler,
                 verbose=1,
                 n_gpu            = None,
                 resume           = None,
                 model_checkpoint = None,
                 training_monitor = None,
                 early_stopping   = None,
                 gradient_accumulation_steps=1):
        self.model            = model              # 模型
        self.train_data       = train_data         # 训练数据
        self.val_data         = val_data           # 验证数据
        self.epochs           = epochs             # epochs次数
        self.optimizer        = optimizer          # 优化器
        self.logger           = logger             # 日志记录器
        self.verbose          = verbose            # 是否打印
        self.training_monitor = training_monitor   # 监控训练过程指标变化
        self.early_stopping   = early_stopping     # early_stopping
        self.resume           = resume             # 是否重载模型
        self.model_checkpoint = model_checkpoint   # 模型保存
        self.evaluate         = evaluate           # 评估指标
        self.criterion        = criterion
        self.lr_scheduler     = lr_scheduler
        self.n_gpu            = n_gpu              # gpu个数，列表形式
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._reset()

    def _reset(self):

        self.batch_num         = len(self.train_data)
        self.progressbar       = ProgressBar(n_batch = self.batch_num,eval_name='acc',loss_name='loss')
        self.model,self.device = model_device(n_gpu=self.n_gpu,model = self.model,logger = self.logger)
        self.start_epoch       = 1
        self.global_step       = 0

        # if self.device == 'cpu':
        #     self.input_type = torch.LongTensor
        # else:
        #     self.input_type = torch.cuda.LongTensor
        # 重载模型，进行训练
        if self.resume:
            arch = self.model_checkpoint.arch
            resume_path = os.path.join(self.model_checkpoint.checkpoint_dir.format(arch = arch),
                                       self.model_checkpoint.best_model_name.format(arch = arch))
            self.logger.info("\nLoading checkpoint: {} ...".format(resume_path))
            resume_list = restore_checkpoint(resume_path = resume_path,model = self.model,optimizer = self.optimizer)
            self.model     = resume_list[0]
            self.optimizer = resume_list[1]
            best           = resume_list[2]
            self.start_epoch = resume_list[3]

            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info("\nCheckpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # for p in model_parameters:
        #     print(p.size())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # 总的模型参数量
        self.logger.info('trainable parameters: {:4}M'.format(params / 1000 / 1000))
        # 模型结构
        self.logger.info(self.model)

    # 保存模型信息
    def _save_info(self,epoch,val_loss):
        state = {
            'epoch': epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': round(val_loss,4)
        }
        return state

    # val数据集预测
    def _valid_epoch(self):
        val_loss,count = 0, 0
        predicts   = []
        targets    = []
        self.model.eval()
        with torch.no_grad():
            for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(self.val_data):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                target = label_ids.to(self.device)
                logits = self.model(input_ids, segment_ids,input_mask)
                loss = self.criterion(target=target, output=logits)
                val_loss += loss.item()
                predicts.append(logits)
                targets.append(target)
                count += 1

            predicts = torch.cat(predicts,dim = 0)
            targets = torch.cat(targets,dim = 0)
            val_acc, val_auc = self.evaluate(output=predicts, target=targets)

        return {
            'val_loss': val_loss / count,
            'val_acc': val_acc,
            'val_auc': val_auc
        }

    # epoch训练
    def _train_epoch(self):
        self.model.train()
        train_loss = AverageMeter()
        train_acc  = AverageMeter()
        for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(self.train_data):

            start = time.time()
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            target = label_ids.to(self.device)

            logits = self.model(input_ids, segment_ids,input_mask)
            loss = self.criterion(output=logits,target=target)
            acc, _ = self.evaluate(output=logits, target=target)

            # 如果梯度更新累加step>1，则也需要进行mean操作
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            loss.backward()
            # 学习率更新方式
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.lr_scheduler.step(training_step = self.global_step)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            train_loss.update(loss.item(),input_ids.size(0))
            train_acc.update(acc,input_ids.size(0))
            if self.verbose >= 1:
                self.progressbar.step(batch_idx= step,
                                      loss     = loss.item(),
                                      acc      = acc,
                                      use_time = time.time() - start)
        print("\ntraining result:")
        train_log = {
            'loss': train_loss.avg,
            'acc': train_acc.avg,
        }
        return train_log

    def train(self):
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):

            print("----------------- training start -----------------------")
            print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch+self.epochs -1))

            train_log = self._train_epoch()
            val_log = self._valid_epoch()

            logs = dict(train_log,**val_log)
            self.logger.info('\nEpoch: %d - loss: %.4f acc: %.4f - val_loss: %.4f - val_acc: %.4f - val_auc: %.4f'%(
                            epoch,logs['loss'],logs['acc'],logs['val_loss'],logs['val_acc'],logs['val_auc']))

            if self.training_monitor:
                self.training_monitor.step(logs)

            if self.model_checkpoint:
                state = self._save_info(epoch,val_loss = logs['val_loss'])
                self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor],state = state)

            if self.early_stopping:
                self.early_stopping.step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break

