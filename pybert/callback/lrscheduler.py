#encoding:utf-8
import numpy as np
import warnings
from torch.optim.optimizer import Optimizer

class StepLr(object):
    def __init__(self,optimizer,lr):
        super(StepLr,self).__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.use = 'on_epoch_end'

    def step(self,epoch):
        lr = self.lr
        if epoch > 12:
            lr = lr / 1000
        elif epoch > 8:
            lr = lr / 100
        elif epoch > 4:
            lr = lr / 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class CustomDecay(object):
    def __init__(self,optimtizer):
        super(CustomDecay,self).__init__()
        self.optimizer = optimtizer

    def step(self,epoch):
        if epoch % 2 != 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group['lr'] * 0.1

class BertLr(object):
    def __init__(self,optimizer,lr,t_total,warmup):
        super(BertLr,self).__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.t_total = t_total
        self.warmup = warmup

    # 线性预热方式
    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def step(self,training_step):
        lr_this_step = self.lr * self.warmup_linear(training_step / self.t_total,self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.


    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_acc, val_loss = validate(...)
        >>>     scheduler.step(val_loss, epoch)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self.use = 'on_epoch_end'
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def reset(self):
        self._reset()

    def step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0
