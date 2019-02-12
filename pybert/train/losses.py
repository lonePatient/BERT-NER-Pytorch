#encoding:utf-8
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

class CrossEntropy(object):
    def __init__(self):
        self.loss_fn = CrossEntropyLoss()
    def __call__(self, output, target):
        loss = self.loss_fn(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        loss = self.loss_fn(input = output,target = target)
        return loss
