#encoding:utf-8
import os
from pybert.config.basic_config import configs as config
from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

if __name__ == "__main__":
    os.system('cp {config} {save_path}'.format(config = config['bert_config_file'],
                                               save_path =config['bert_model_dir']))
    convert_tf_checkpoint_to_pytorch(config['tf_checkpoint_path'],
                                     config['bert_config_file'],
                                     config['pytorch_model_path'])
