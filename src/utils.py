#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:utlis.py
@time:2021/09/09
"""
import binascii

import yaml
import requests
import sys, os, re
import logging
import uuid
import datetime
import cv2, base64
import numpy as np
import time

def read_yaml(config_path):
    """
    yaml 配置文件
    :param config_path:
    :return: 
    """
    config_reader = open(config_path, 'r', encoding='utf-8')
    cfg = config_reader.read()
    config_file = yaml.load(cfg, Loader=yaml.FullLoader)

    return config_file


def download_img(img_url, save_img_name):
    """
    根据url 下载图片，并且按照指定的文件保存图像
    :param img_url:
    :param save_img_name:
    :return:
    """
    try:
        # 图片下载异常捕捉
        r = requests.get(img_url)
    except requests.exceptions.ConnectionError as e:
        logging.info("Download images exception %s ", e)
        return False, "Download images exception ! "

    if r.status_code == 200:
        img_mb = sys.getsizeof(r.content) / 1038336
        logging.info('url image is %s MB', img_mb)
        with open(save_img_name, 'wb')as f:
            # 将内容写入图片
            f.write(r.content)

        return True, img_mb

    del r
    return False, "Download fail !"


def send_dd_warning(info,url='https://oapi.dingtalk.com/robot/send?access_token=b76221c46710911949534c39ec6e5e889463dae332109b18adcc1170495c6024'):
    """
    向钉钉发送警告
    Args:
        info: 警告信息

    Returns:

    """
    # url = 'https://oapi.dingtalk.com/robot/send?access_token=3a625662e46486d284c5f7f211112304404c108e64610fe264b632b2b6209b51'
    message = {"msgtype": "text", "text": {"content": info}}
    try:
        res = requests.post(url, json=message)
        logging.info('钉钉警告发送成功!')
    except requests.exceptions.ConnectionError as e:
        logging.info('钉钉警告发送失败!')


def get_image_name():
    img_id = str(uuid.uuid1())
    img_name = img_id + ".jpg"

    now = datetime.datetime.now()
    create_time = now.strftime("%Y-%m-%d %H:%M:%S")

    dataset_dir = './img'
    time_dir = os.path.join(dataset_dir, str(now.strftime("%Y-%m-%d")))
    # date as save image dir
    if not os.path.exists(time_dir):
        os.makedirs(time_dir)
    # save image on year-month-day
    save_img_name = time_dir + "/" + img_name
    return save_img_name


def base64_to_cv_image(img_base64, save_img_name):
    ##data:image/jpeg;base64,
    reg = r'^data:[a-z]+/[a-z]+;base64,'
    tag = re.findall(reg, img_base64)

    if len(tag) == 1:
        img_base64 = img_base64.replace(tag[0], "")
        logging.info("new img_base64 %s", tag)

    # 1. save request info
    # 1M = 1024*1024 = 1038336
    img_mb = sys.getsizeof(img_base64) / 1038336
    logging.info("img_mb : {}".format(img_mb))

    # 限定 12 M
    if img_mb > 10:
        logging.info("img > 12 M")
        return False, "img is larger than 12 M"

    # decode image base64

    try:
        npstr = base64.b64decode(img_base64)

        img_array = np.frombuffer(npstr, np.uint8)
        nparr_re = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

        logging.info("Decode request image and save !")
        cv2.imwrite(save_img_name, nparr_re)
        return True, 'success'
        # return jsonify({'code': 'E001', 'message': 'make_response', 'success': False})
    except binascii.Error:
        return False, "base64 bytes exception"
    except cv2.error as e:
        logging.info("cv2 exception %s", e)


        return False, "base64 image exception"


def check_image_request_params(request_data_dict, index_params):
    logging.info("Load Json Data ! ")
    # 参数异常处理
    for key in index_params:
        if key not in request_data_dict.keys():
            return None
        # 如果存在，值不能为空
        else:
            value = request_data_dict[key]
            # 空值或空字符串
            if value is None or value is '':
                return None

    return request_data_dict


def get_img(params_data,img_name):
    """
    从请求参数中获取图片并返回
    :param params_data:
    :return:
    """
    image = params_data['image']
    image_type = params_data['image_type']



    # 暂时只支持url,base64
    if image_type == "URL":
        logging.info("download image {}".format(image))

        return download_img(image, img_name)

    elif image_type == 'BASE64':

        return base64_to_cv_image(image, img_name)

    else:

        return False, 'image type not support !!!'

def get_upload_time(params_data):

    if 'upload_time' in params_data.keys():
        upload_time = params_data['upload_time']
    else:
        # 秒级时间戳
        upload_time = time.time()


    return int(upload_time)
