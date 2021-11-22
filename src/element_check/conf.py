# coding: utf-8

import yaml
from src.utils import read_yaml


class LabelConf(object):
    def __init__(self, conf_path):
        # self.conf = None
        # with open(conf_path, 'r') as f:
        #     self.conf = yaml.safe_load(f.read())
        self.conf = read_yaml(conf_path)

    def confidence_filter(self, tuples):
        """ 置信度过滤 """
        cf = self.conf['label_confidence']
        tuples = filter(lambda v: v.confidence >= cf[v.label], tuples)
        return list(tuples)

    def get_labels(self):
        """ 获取标签名称 """
        return self.conf['labels']

    def get_label_template(self):
        return self.conf['label_temp']

    def get_time_template(self):
        return self.conf['time_temp']
