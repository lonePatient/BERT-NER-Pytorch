#encoding:utf-8
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import roc_auc_score,accuracy_score

class Accuracy(object):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    '''
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK

    def __call__(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topK].view(-1).float().sum(0)
        result = correct_k / batch_size
        return result

# auc
class AUCThresh(object):
    def __init__(self,thresh,sigmoid):
        self.thresh = thresh
        self.sigmoid = sigmoid

    def __call__(self,output,target):
        target = target.cpu().numpy()
        if self.sigmoid:
            y_pred = output.sigmoid().cpu().numpy()
        else:
            y_pred = output.cpu().numpy()
        auc = roc_auc_score(y_score=y_pred,y_true=target,average="macro")
        acc = np.mean(((y_pred > self.thresh) == target).astype(int)).sum()
        return acc,auc

# 計算F1得分
class F1Score(object):
    def __init__(self):
        pass
    def __call__(self,output,target):
        _, y_pred = torch.max(output.data, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = target.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        return (acc, f1)

# 多类别分类报告
class ClassReport(object):
    def __init__(self,target_names = None):
        self.target_names = target_names

    def __call__(self,output,target):
        _, y_pred = torch.max(output.data, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = target.cpu().numpy()
        classify_report = classification_report(y_true, y_pred,target_names=self.target_names)
        print('\n\nclassify_report:\n', classify_report)
