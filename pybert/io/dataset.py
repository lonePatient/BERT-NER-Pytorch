#encoding:utf-8
import csv
import numpy as np
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid   = guid  # 该样本的唯一ID
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeature(object):
    '''
    数据的feature集合
    '''
    def __init__(self,input_ids,input_mask,segment_ids,label_id,output_mask):
        self.input_ids   = input_ids   # tokens的索引
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id
        self.output_mask = output_mask

class CreateDataset(Dataset):
    def __init__(self,data_path,max_seq_len,vocab_path,example_type,seed):
        self.seed    = seed
        self.max_seq_len  = max_seq_len
        self.example_type = example_type
        self.data_path  = data_path
        self.vocab_path = vocab_path
        self.reset()

    # 初始化
    def reset(self):

        # 加载语料库，这是pretrained Bert模型自带的
        self.tokenizer = BertTokenizer(vocab_file=self.vocab_path)
        # 构建examples
        self.build_examples()

    # 读取数据集
    def read_data(self,quotechar = None):
        '''
        默认是以tab分割的数据
        :param quotechar:
        :return:
        '''
        lines = []
        with open(self.data_path,'r',encoding='utf-8') as fr:
            reader = csv.reader(fr,delimiter = '\t',quotechar = quotechar)
            for line in reader:
                lines.append(line)
        return lines

    # 构建数据examples
    def build_examples(self):
        lines = self.read_data()
        self.examples = []
        for i,line in enumerate(lines):
            guid = '%s-%d'%(self.example_type,i)
            label = [int(x) for x in line[0].split(',')]
            text_a = line[1]
            example = InputExample(guid = guid,text_a = text_a,label= label)
            self.examples.append(example)
        del lines

    # 将example转化为feature
    def build_features(self,example):
        '''
        # 对于两个句子:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1

        # 对于单个句子:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # type_ids:表示是第一个句子还是第二个句子
        '''
        # load sub_vocab
        sub_vocab = {}
        with open(self.VOCAB_FILE, 'r') as fr:
            for line in fr:
                _line = line.strip('\n')
                if "##" in _line and sub_vocab.get(_line) is None:
                    sub_vocab[_line] = 1

        #转化为token
        tokens_a = self.tokenizer.tokenize(example.text_a)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.max_seq_len - 2:
            tokens_a = tokens_a[:(self.max_seq_len - 2)]
        # 句子首尾加入标示符
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)  # 对应type_ids
        # 将词转化为语料库中对应的id
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # 输入mask
        input_mask = [1] * len(input_ids)
        # padding，使用0进行填充
        padding = [0] * (self.max_seq_len - len(input_ids))

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        # ---------------处理target----------------
        ## Notes: label_id中不包括[CLS]和[SEP]
        label_id = [label_map[l] for l in labels]
        label_padding = [-1] * (max_seq_length-len(label_id))
        label_id += label_padding
        ## output_mask用来过滤bert输出中sub_word的输出,只保留单词的第一个输出(As recommended by jocob in his paper)
        ## 此外，也是为了适应crf
        output_mask = [0 if sub_vocab.get(t) is not None else 1 for t in tokens_a]
        output_mask = [0] + output_mask + [0]
        output_mask += padding

        # ----------------处理后结果-------------------------
        # for example, in the case of max_seq_length=10:
        # raw_data:          春 秋 忽 代 谢le
        # token:       [CLS] 春 秋 忽 代 谢 ##le [SEP]
        # input_ids:     101 2  12 13 16 14 15   102   0 0 0
        # input_mask:      1 1  1  1  1  1   1     1   0 0 0
        # label_id:          T  T  O  O  O
        # output_mask:     0 1  1  1  1  1   0     0   0 0 0
        # --------------看结果是否合理------------------------

        feature = InputFeature(input_ids   = input_ids,input_mask = input_mask,
                               segment_ids = segment_ids,label_id = label_id,
                               output_mask = output_mask)
        return feature

    def _preprocess(self,index):
        example = self.examples[index]
        feature = self.build_features(example)
        return np.array(feature.input_ids),np.array(feature.input_mask),\
               np.array(feature.segment_ids),np.array(feature.label_id),\
               np.array(feature.output_mask)

    def __getitem__(self, index):
        return self._preprocess(index)

    def __len__(self):
        return len(self.examples)
