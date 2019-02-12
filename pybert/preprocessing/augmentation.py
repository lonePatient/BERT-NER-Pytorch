#encoding:utf-8
import numpy as np
import random

class Augmentator(object):
    def __init__(self,is_train_mode = True, proba = 0.5):
        self.mode = is_train_mode
        self.proba = proba
        self.augs = []
        self._reset()

    # 总的增强列表
    def _reset(self):
        self.augs.append(lambda text: self._shuffle(text))
        self.augs.append(lambda text: self._dropout(text,p = 0.5))

    # 打乱
    def _shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    #随机删除一些
    def _dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)

    def __call__(self,text,aug_type):
        '''
        用aug_type区分数据
        '''
        # TTA模式
        if 0 <= aug_type <= 2:
            pass
        # 训练模式
        if self.mode and  random.random() < self.proba:
            aug = random.choice(self.augs)
            text = aug(text)
        return text
