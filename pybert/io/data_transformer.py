#encoding:utf-8
import random
from tqdm import tqdm
from ..utils.utils import text_write

class DataTransformer(object):
    def __init__(self,
                 logger,
                 label_to_id,
                 train_file,
                 valid_file,
                 valid_size,
                 skip_header,
                 preprocess,
                 raw_data_path,
                 raw_target_path,
                 shuffle,
                 stratify,
                 seed,
                 ):
        self.seed          = seed
        self.logger        = logger
        self.valid_size    = valid_size
        self.train_file    = train_file
        self.valid_file    = valid_file
        self.raw_data_path = raw_data_path
        self.raw_target_path = raw_target_path
        self.skip_header   = skip_header
        self.label_to_id   = label_to_id
        self.preprocess    = preprocess
        self.shuffle       = shuffle
        self.stratify      = stratify

    # 分割
    def split_sent(self, line):
        res = line.strip('\n').split()
        return res

    # 将原始数据集分割成train和valid
    def train_val_split(self,X, y,stratify=False):
        self.logger.info('train val split')
        if stratify:
            train, valid = [], []
            bucket = [[] for _ in self.label_to_id]
            for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
                bucket[int(data_y)].append((data_x, data_y))
            del X, y
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * self.valid_size)
                if self.shuffle:
                    random.seed(self.seed)
                    random.shuffle(bt)
                valid.extend(bt[:test_size])
                train.extend(bt[test_size:])
            # 混洗train数据集
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(train)
            return train, valid
        else:
            data = []
            for data_x, data_y in tqdm(zip(X, y), desc='Merge'):
                data.append((data_x, data_y))
            del X, y
            N = len(data)
            test_size = int(N * self.valid_size)
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(data)
            valid = data[:test_size]
            train = data[test_size:]
            return train, valid

    # 读取原始数据集
    def read_data(self):
        targets,sentences = [],[]
        with open(self.raw_data_path,'r') as fr_x,open(self.raw_target_path,'r') as fr_y:
            sentences, labels = [], []
            for i, (sent, target) in enumerate(zip(fr_x, fr_y)):
                if i == 0 and self.skip_header:
                    continue
                words = self.split_sent(sent)
                label = self.split_sent(target)
                if len(words) == 0 or len(label) == 0:
                    continue
                sentences.append(words)
                labels.append(label)
        # 保存数据
        if self.valid_size:
            train,valid = self.train_val_split(X = sentences,y = targets,stratify=self.stratify)
            text_write(filename = self.train_file,data = train)
            text_write(filename = self.valid_file,data = valid)
        else:
            pass


