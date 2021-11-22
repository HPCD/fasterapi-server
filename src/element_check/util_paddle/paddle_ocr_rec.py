#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:paddle_ocr_rec.py
@time:2021/11/03
"""
import os
import sys
import subprocess
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, 'ocr_infer')))
sys.path.append(os.path.abspath(os.path.join(__dir__, 'ocr_infer', 'ppocr')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
import logging
from PIL import Image
import utility as utility
import predict_rec as predict_rec
from predict_rec import TextRecognizer


args = utility.parse_args()
text_recognizer = TextRecognizer(args)

def get_rec_result(img):
    """
    Args:
        img_path: 图片路径

    Returns: tuple (识别字符，得分) 如：（‘123’，0.9），空返回：（‘’，nan）

    """
    # img = cv2.imread(img_path)

    rec_res, _ = text_recognizer([img])
    logging.info("rec:{} ".format(rec_res[0]))
    return rec_res


def extract_text(img):
    rec_res = get_rec_result(img)
    ret = []
    for v in rec_res:
        if v[0]:
            data = {
                'text': v[0],
                'confidence': v[1],
                'cx': 0.5,
                'cy': 0.5
            }
            ret.append(data)
    return ret

if __name__ == "__main__":
    img_path = r'E:\dataset\friends\wandou\images\id_images\0_0_0a3f9df8be6946cbb210a50cc3012d18.png'
    get_rec_result(img_path)