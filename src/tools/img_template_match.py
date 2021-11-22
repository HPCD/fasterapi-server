#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:img_template_match.py
@time:2021/09/28
"""
import logging

import cv2
import numpy as np
import os


def match_template(img_path, temp_conf):

    temp_poster_in = temp_conf['poster']['include']
    temp_poster_ex = temp_conf['poster']['exclude']
    temp_logo_in = temp_conf['logo']['include']
    temp_logo_ex = temp_conf['logo']['exclude']
    if len(temp_logo_in) > 0:
        for template_path in temp_logo_in:
            if multiscale_template(img_path,template_path):
                os.remove(img_path)
                return True
    os.remove(img_path)
    return False


def multiscale_template(img_path,template_path):
    global img_w, img_h
    # src_img_path = r'E:\dataset\friends\hualala\5.png'
    logging.info("img_path: {}".format(img_path))
    logging.info("template_path: {}".format(template_path))
    img_rgb = cv2.imread(img_path)
    img = cv2.imread(img_path, 0)
    img_w, img_h = img.shape[::-1]

    template = cv2.imread(template_path,0)
    min_zoom_ratio = 0.02
    ratio_gap = 100

    match_result = []
    match_result_dict = {}
    temp_size = {}

    for scale in np.linspace(min_zoom_ratio, 1.0, ratio_gap):
        # template_resized = imutils.resize(template, width=int(template.shape[1] * scale))
        template_resized = cv2.resize(template, None, fx=scale, fy=scale)

        if template_resized.shape[0] > img_h or template_resized.shape[1] > img_w:
            continue
        target = img.copy()
        # 获得模板图片的高宽尺寸
        theight, twidth = template_resized.shape[:2]
        # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(target, template_resized,cv2.TM_SQDIFF_NORMED)
        threshold = 0.8
        # loc = np.where(result >= threshold)

        cv2.waitKey(0)
        # 归一化处理
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
        # 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # 匹配值转换为字符串
        # 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
        # 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc

        # 绘制矩形边框，将匹配区域标注出来
        # min_loc：矩形定点
        # (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
        # (0,0,225)：矩形的边框颜色；2：矩形边框宽度
        new_img = img_rgb.copy()
        # cv2.rectangle(new_img, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)

        cut_img = img[int(min_loc[1]):int(min_loc[1] + theight), min_loc[0]:int(min_loc[0] + twidth)]

        dis = img_similary(cut_img, template_resized)
        # print("dis : ",dis)

        if dis < 5:
            logging.info("logo match dis {}".format(dis))
            return True

    return False



def img_similary(img1, img2):
    """
    图片相似度
    Args:
        img1: 图片 np.array
        img2: 图片 np.array

    Returns:

    """
    img1_hash = img_hash(img1)
    img2_hash = img_hash(img2)
    dis = hamming_distance(img1_hash, img2_hash)
    # print('hanming di:{} '.format(dis))
    return dis


def hamming_distance(hash1, hash2):
    """汉明距离"""
    count = 0
    for i in range(0, len(hash1)):
        if hash1[i] != hash2[i]:
            count += 1
    return count


def img_hash(img):
    """
    图片哈希值
    Args:
        img:

    Returns:

    """
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC
    # 转换为灰度图
    try:
        gray = img.copy()
    except:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash
