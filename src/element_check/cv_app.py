# coding: utf-8

import sys
import re
import cv2
import base64
import binascii
import numpy as np
from collections import namedtuple


KERNEL_SIZE = 9


def show_img(title, img, resize=1):
    h, w = img.shape[:2]
    rh, rw = int(h * resize), int(w * resize)
    cv2.imshow(title, cv2.resize(img, (rw, rh)))

def wait_img():
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def encode_image(img):
    img_encode = cv2.imencode('.jpg', img)[1]
    return str(base64.b64encode(img_encode))[2:-1]


def decode_image(img_base64):
    ##data:image/jpeg;base64,
    reg = r'^data:[a-z]+/[a-z]+;base64,'
    tag = re.findall(reg, img_base64)

    if len(tag) == 1:
        img_base64 = img_base64.replace(tag[0], "")
        print("new img_base64 %s" % tag)

    # 1. save request info
    # 1M = 1024*1024 = 1038336
    img_mb = sys.getsizeof(img_base64) / 1038336
    print("img_mb : {}".format(img_mb))

    # 限定 10 M
    if img_mb > 10:
        print("img is larger than 10 M")
        return None

    # decode image base64
    try:
        npstr = base64.b64decode(img_base64)
        img_array = np.frombuffer(npstr, np.uint8)
        return cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    except binascii.Error as e:
        print("base64 exception %s" % e, file=sys.stderr)
        return None
    except cv2.error as e:
        print("cv2 exception %s" % e, file=sys.stderr)
    
    return None


def gray(img):
    """ 灰度处理 """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def blur(img):
    """ 滤波 """
    return cv2.medianBlur(img, KERNEL_SIZE)


def threshold(img, thres1=250, thres2=50):
    """ 二值化 """
    _, img_thre = cv2.threshold(img, thres1, 255, cv2.THRESH_BINARY)
    inv = False
    # 直方图
    img_hist = cv2.calcHist([img_thre], [0], None, [256], [0, 256])
    img_hist = np.squeeze(img_hist)
    black_rate = sum(img_hist[:127]) / sum(img_hist)
    white_rate = sum(img_hist[127:]) / sum(img_hist)
    # 如果几乎都是黑色, 说明是黑色背景, 需要反向处理
    if black_rate >= 0.9:
        _, img_thre = cv2.threshold(img, thres2, 255, cv2.THRESH_BINARY_INV)
        inv = True
    return img_thre, inv


def edge(img):
    """ 边缘检测 """
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)
    wm = np.sqrt(np.square(gx) + np.square(gy))
    return cv2.normalize(wm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


def contour(img):
    """ 搜索轮廓 """
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contour2rect(contours):
    """ 轮廓矩形化 """
    rects = []
    for con in contours:
        x, y, width, height = cv2.boundingRect(con)
        rects.append(
            namedtuple('Rect', 'x y w h')(
                x, y, width, height
            )
        )
    return rects


def is_back_black(img):
    """ 检测背景是否是黑色 """
    img = gray(img)
    # img = blur(img)
    img_thre, inv = threshold(img)
    return inv


def search_rect(img0, space=0, fill=(0,0), show=False):
    """ 搜索矩形区域 """
    img = gray(img0)
    if show:
        show_img('gray', img)

    if space > 0:
        h0, w0 = img0.shape[:2]
        c = img[fill]
        img_new = np.zeros((h0+2*space, w0+2*space), dtype=np.uint8)
        img_new[space:space+h0, space:space+w0] = img
        img_new[0:space,:] = c
        img_new[space+h0:,:] = c
        img_new[:,0:space] = c
        img_new[:,space+w0:] = c
        img = img_new
        if show:
            show_img('space', img)

    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # if show:
    #     show_img('gauss', img)

    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # if show:
    #     show_img('thres', img)

    img = edge(img)
    img = 255 - img
    img[img < 250] = 0
    img[img >= 250] = 255
    img = 255 - img
    if show:
        show_img('edge', img)

    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = contour2rect(contours)
    rects_new = []
    for r in rects:
        rects_new.append(namedtuple('Rect', 'x y w h')(
            r.x-space,
            r.y-space,
            r.w, r.h
        ))
    rects = rects_new
    if show:
        img_cp = img0.copy()
        for r in rects:
            cv2.rectangle(img_cp, (r.x, r.y), (r.x+r.w, r.y+r.h), (0,255,0), 2)
        show_img('contour', img_cp)

    return rects


import src.element_check.label as lb
def search_rect_filter(rects, cut=0.5):
    """ 矩形区域搜索结果过滤 """
    if len(rects) <= 1:
        return rects
    # 过滤太小的
    rects = sorted(rects, key=lambda v: v.w*v.h, reverse=True)
    for i in range(len(rects)-1):
        r1 = rects[i]
        r2 = rects[i+1]
        s1 = r1.w * r1.h
        s2 = r2.w * r2.h
        if s2 < cut * s1:
            break
    rects = rects[0:i+1]
    # 过滤重复的
    arr1 = []
    for i in range(len(rects)):
        r1 = rects[i]
        r11 = namedtuple('Rect', 'cx, cy, width, height')(
            r1.x + 0.5 * r1.w,
            r1.y + 0.5 * r1.h,
            r1.w, r1.h
        )
        arr2 = [r1]
        for j in range(i+1, len(rects)):
            r2 = rects[j]
            r22 = namedtuple('Rect', 'cx, cy, width, height')(
                r2.x + 0.5 * r2.w,
                r2.y + 0.5 * r2.h,
                r2.w, r2.h
            )
            iou_v = lb.LabelTuple.iou_ver(r11, r22)
            iou_h = lb.LabelTuple.iou_hor(r11, r22)
            if iou_v > 0.5 and iou_h > 0.5:
                arr2.append(r2)
        arr1.append(arr2)
    rects = []
    for arr2 in arr1:
        rects.append(
            max(arr2, key=lambda v: v.w*v.h)
        )
    return rects


def expand(img, ratio=(1.5, 1.5), fill=255):
    """ 扩展图片 """
    assert ratio[0] >= 1 and ratio[1] >= 1
    h0, w0 = img.shape[:2]
    w = int(w0 * ratio[0])
    h = int(h0 * ratio[1])
    img_new = np.zeros((h, w, 3), np.uint8)
    img_new[:,:,:] = fill
    x_start = int(0.5 * (w - w0))
    y_start = int(0.5 * (h - h0))
    img_new[y_start:y_start+h0, x_start:x_start+w0, :] = img
    return img_new


def separate(img, ws, dir='hor', reverse=False, img_wb=None):
    """
    扫描空白区域并割开图片(取最空白的区域作为分割线)
    img: 图片
    ws: 扫描窗口宽度
    dir: 扫描方向(hor: 从左至右, ver: 从上至下)
    reverse: 是否反向扫描
    img_wb: 判断背景是否是黑色还是白色的图片
    """
    assert dir in ['hor', 'ver']
    is_hor = dir == 'hor'
    h, w = img.shape[:2]
    # 是否是黑色背景 
    black = is_back_black(img) if img_wb is None else is_back_black(img_wb)
    # 灰度处理
    img_gray = gray(img)
    # 扫描方向
    start = 0
    end = (w - ws) if is_hor else (h - ws)
    step = 1
    if reverse:
        st = start
        start = end
        end = st
        step = -1
    # 扫描, 找出分割中线
    a = []
    for i in range(start, end, step):
        img_scan = img_gray[:, i:i+ws] if is_hor else img_gray[i:i+ws, :]
        a.append((i, np.sum(img_scan)))
    ma = min(a, key=lambda v: v[1]) if black else max(a, key=lambda v: v[1])
    si = ma[0] + int(0.5 * ws)
    return si / (w if is_hor else h)


def separate2(img, ws, dir='hor', reverse=False, img_wb=None, thres=0.95):
    """
    扫描空白区域并割开图片(取第一个相对空白的区域作为分割线)
    img: 图片
    ws: 扫描窗口宽度
    dir: 扫描方向(hor: 从左至右, ver: 从上至下)
    reverse: 是否反向扫描
    img_wb: 判断背景是否是黑色还是白色的图片
    """
    assert dir in ['hor', 'ver']
    is_hor = dir == 'hor'
    h, w = img.shape[:2]
    # 是否是黑色背景 
    black = is_back_black(img) if img_wb is None else is_back_black(img_wb)
    # 灰度处理
    img_gray = gray(img)
    if black:
        img_gray[img_gray < 50] = 0
    else:
        img_gray[img_gray > 250] = 255
    # 扫描方向
    start = 0
    end = (w - ws) if is_hor else (h - ws)
    step = 1
    if reverse:
        st = start
        start = end
        end = st
        step = -1
    # 扫描, 找出分割中线
    t = 255 * h * ws
    for i in range(start, end, step):
        img_scan = img_gray[:, i:i+ws] if is_hor else img_gray[i:i+ws, :]
        s = (t - np.sum(img_scan)) if black else np.sum(img_scan)
        if s / t > thres:
            si = i + int(0.5 * ws)
            return si / (w if is_hor else h)
    return None
    


if __name__ == '__main__':
    img_path = '/Users/penghuan/Desktop/000.jpg'
    img = cv2.imread(img_path)
    search_rect(img, space=50, fill=(5,5), show=True)
    wait_img()
