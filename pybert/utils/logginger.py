#encoding:utf-8
import os
import logging
from logging import Logger
from .utils import ensure_dir
from logging.handlers import TimedRotatingFileHandler

'''
日志模块
1. 同时将日志打印到屏幕跟文件中
2. 默认值保留近30天日志文件
'''

def init_logger(log_name,log_dir):
    ensure_dir(log_dir)
    if log_name not in Logger.manager.loggerDict:
        logger  = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        handler = TimedRotatingFileHandler(filename=os.path.join(log_dir,"%s.log"%log_name),when='D',backupCount = 30)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str,datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        console= logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        handler = TimedRotatingFileHandler(filename=os.path.join(log_dir,"ERROR.log"),when='D',backupCount= 30)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str,datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
    logger = logging.getLogger(log_name)
    return logger

if __name__ == "__main__":
    logger = init_logger('test','D:\data')
    logger.info("test")
    logger.error("test2")