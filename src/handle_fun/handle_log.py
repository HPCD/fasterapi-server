#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:handle_log.py
@time:2021/09/09
"""
import re
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import py_eureka_client.logger as eureka_logger

def log(log_path='logs/',log_name="/wechat-moments-verify.txt"):
    """
     save logs file and console print
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    fh = TimedRotatingFileHandler(filename=log_path + log_name, when="D", interval=1, backupCount=7)
    fh.suffix = "%Y-%m-%d_%H-%M.logs"
    fh.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.logs$")
    ch = logging.StreamHandler()
    formatter = logging.Formatter(('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 文件输出
    logger.addHandler(fh)
    # 控制台输入
    logger.addHandler(ch)

    # eureka logs

    _formatter = logging.Formatter(fmt='[%(asctime)s]-[%(name)s]-%(levelname)-4s: %(message)s')
    _handler = TimedRotatingFileHandler(log_path + "/py-eureka-client.logs", when="midnight", backupCount=7)
    _handler.setFormatter(_formatter)
    _handler.setLevel("INFO")

    eureka_logger.set_handler(_handler)

    return logger

