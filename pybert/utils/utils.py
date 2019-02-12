#encoding:utf-8
import os
import random
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm

# 设置seed环境
def seed_everything(seed = 1029,device='cpu'):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# batch的数据处理
def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    transposed = zip(*batch)
    lbd = lambda batch:torch.cat([torch.from_numpy(b).long() for b in batch])
    return [lbd(samples) for samples in transposed]

class AverageMeter(object):
    '''
    computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n = 1):
        self.val  = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def json_write(data,filename):
    with open(filename,'w') as f:
        json.dump(data,f)

def json_read(filename):
    with open(filename,'r') as f:
        return json.load(f)

def pkl_read(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def pkl_write(filename,data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def text_write(filename,data):
    with open(filename,'w') as fw:
        for sentence,target in tqdm(data,desc = 'write data to disk'):
            sentence = [str(x) for x in sentence]
            target  = [str(x) for x in target]
            line = '\t'.join([" ".join(target),' '.join(sentence)])
            fw.write(line +'\n')