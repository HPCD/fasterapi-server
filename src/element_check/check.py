# coding: utf-8

import os
import re
import sys
import cv2
import time
import json
import numpy as np
from itertools import chain
from collections import namedtuple
from urllib.request import Request, urlopen
from urllib.error import URLError

import src.element_check.cv_app as cva
import src.element_check.label as lb
from src.element_check.conf import LabelConf
from src.element_check.yolov5.yolo_model import YoloModel
from src.element_check.util_paddle.paddle_ocr import extract_text as pd_extract_text
from src.element_check.util_easyocr import extract_text as es_extract_text
from src.element_check.util_paddle.paddle_ocr_rec import extract_text as pd_extract_text_mini
from src.tools.chinese_font_conversion import cht_to_chs
from src.element_check.qrcode import WorkerQrPool


DIR_TMP = 'img'
YOLO_MODEL_PATH = os.path.join('models', 'moment_check_20211112.pt')
LABEL_CONF_PATH = os.path.join('config', 'moment_label.yaml')
QR_PROTO_DT_PATH = os.path.join('models', 'detect.prototxt')
QR_PROTO_SR_PATH = os.path.join('models', 'sr.prototxt')
QR_MODEL_DT_PATH = os.path.join('models', 'detect.caffemodel')
QR_MODEL_SR_PATH = os.path.join('models', 'sr.caffemodel')

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0


def download(url, to):
    try:
        request = Request(url)
        response = urlopen(request)
        raw = response.read()
        with open(to, 'wb') as f:
            f.write(raw)
        print('Downloaded from %s, save to %s' % (url, to,))
        return True
    except Exception as err:
        print('Download Fail, %s, %s' % (err, url,), file=sys.stderr)
    return False


class LabelChecker(object):
    """ 标签检测 """
    def __init__(self, checker_id):
        self.checker_id = checker_id
        self.yolo_model = YoloModel(YOLO_MODEL_PATH)
        self.label_conf = LabelConf(LABEL_CONF_PATH)
        # self.qr_pool = WorkerQrPool(1, 4)
        self.qr_parser = cv2.wechat_qrcode_WeChatQRCode(
            QR_PROTO_DT_PATH,
            QR_MODEL_DT_PATH,
            QR_PROTO_SR_PATH,
            QR_MODEL_SR_PATH
        )

    def match(self, img_source, ts=None, debug=False):
        """
        匹配图片各个元素
        img_source: 图片源
        ts: 图片上传时间戳
        """
        # 初始化参数
        self.time_anchor = int(time.time()) if ts is None else ts
        self.img_name = re.split(r'/|\\', img_source)[-1]
        self.img_type = self.img_name.split('.')[-1]
        if self.img_type.lower() not in ['jpg', 'jpeg', 'png']:
            self.img_type = 'jpg'
        self.img_tmp_name = '%s_%03d' % (self.__class__.__name__, self.checker_id)
        self.img_tmp_path = os.path.join(DIR_TMP, '%s.%s' % (self.img_tmp_name, self.img_type,))

        # 下载图片
        if img_source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://')
        ):
            download(img_source, self.img_tmp_path)
        print('img path ',self.img_tmp_path)
        # 加载图片
        self.yolo_model.load_img(self.img_tmp_path)
        self.yolo_model.save_img(self.img_tmp_path)
        self.yolo_model.load_img(self.img_tmp_path)

        # 识别元素
        preds = self.yolo_model.pred(conf_thres=0.1)
        # print(preds)
        if len(preds) == 0:
            self.screen = -2
            self.matched_groups = []
            return

        # 构造标签元组
        tuples = list(map(lambda v: lb.LabelTuple(**v), preds))
        tuples = self.label_conf.confidence_filter(tuples)
        self.tuples = tuples
        self.label_group = lb.LabelGroup(tuples)

        # 识别文字
        self.sentences, self.words = self.parse_text(debug=debug)
        
        # 匹配模版
        self.screen, self.matched_groups = self.label_group.match_temp_most(self.label_conf.get_label_template())

        # 初始化各元素
        tuples_sorted = sorted(tuples, key=lambda v: v.cy)
        for g in self.matched_groups:
            for e in lb.ELE_FILTER_ALL:
                g.elements[e] = lb.ElementNone(name=e)
            # 底部区域
            r = g.get_rect()
            t = None
            cx = -1
            cf = -1
            for _t in tuples_sorted:
                if _t.label == lb.LABEL_LS or _t.label == lb.LABEL_CS:
                    if _t.confidence > cf:
                        cx = _t.cx
                        cf = _t.confidence
            for _t in tuples_sorted:
                if not r.inside(_t.cx, _t.cy) \
                   and _t.cy > r.cy \
                   and _t.label == lb.LABEL_PT_NN \
                   and _t.inside_ver(cx if cx > 0 else _t.cx, _t.cy) \
                :
                    v = True
                    for _t2 in tuples_sorted:
                        if _t2.label == lb.LABEL_LS or _t2.label == lb.LABEL_CS:
                            if _t.inside(_t2.cx, _t2.cy):
                                v = False
                                break
                    if v:
                        t = _t
                        break
            if t is not None:
                g.bottom = lb.Rect(
                    r.cx,
                    0.5 * (r.y2 + t.y1),
                    r.width,
                    t.y1 - r.y2
                )
            else:
                g.bottom = lb.Rect(
                    r.cx,
                    0.5 * (r.y2 + 1.0),
                    r.width,
                    1.0 - r.y2
                )

    def crop_img(self, element):
        """ 裁剪图片 """
        return self.yolo_model.crop_img(**element.asdict())

    def parse_text(self, debug=False):
        """ 解析文本 """
        shape = self.yolo_model.get_img_shape()
        w, h = shape['width'], shape['height']
        # 解析文本
        sentences = pd_extract_text(self.yolo_model.img_ori)
        # 按照行列重排序
        sentences = sorted(sentences, key=lambda v: v['cy'])
        lines = []
        aline = []
        head = sentences[0] if len(sentences) > 0 else None
        for v in sentences:
            if abs(head['cy'] - v['cy']) <= 1:
                aline.append(v)
            else:
                lines.append(aline)
                aline = [v]
                head = v
        if len(aline) > 0:
            lines.append(aline)
            aline = []
        lines2 = []
        for aline in lines:
            aline = sorted(aline, key=lambda v: v['cx'])
            lines2.append(aline)
        sentences = list(chain(*lines2))
        # 坐标格式化
        def fmt(v):
            v['cx'] = v['cx'] / w
            v['cy'] = v['cy'] / h
            return v
        sentences = list(map(fmt, sentences))
        words = []
        for v in sentences:
            for w in chain(*v['text']):
                words.append({'word': w, 'cx': v['cx'], 'cy': v['cy']})
        if debug:
            print('*****************************************************')
            print('\n'.join(map(lambda v: '%s, cx:%.2f, cy:%.2f, cf:%.2f' % (v['text'], v['cx'], v['cy'], v['confidence']), sentences)))
            print('*****************************************************')
        return sentences, words

    def get_text(self, element):
        """ 获取文本信息 """
        return ''.join(map(
            lambda v: v['text'], 
            filter(lambda v: element.inside(v['cx'], v['cy']), self.sentences)
        ))

    def get_text_hor(self, element):
        """ 获取文本信息 """
        return ''.join(map(
            lambda v: v['text'], 
            filter(lambda v: element.inside_hor(v['cx'], v['cy']), self.sentences)
        ))

    def get_text_all(self):
        """ 获取文本信息 """
        return ''.join(map(
            lambda v: v['text'], self.sentences
        ))

    def get_text_pos(self, s, e):
        """ 获取文本位置 """
        return list(map(
            lambda v: (v['cx'], v['cy']), self.words[s:e]
        ))

    def ex_twodot(self, group):
        """ 两个点按钮 """
        t = group.search_label(lb.LABEL_TD)
        if t is not None:
            group.elements[lb.ELEMENT_TD] = lb.ElementImg(
                name=lb.ELEMENT_TD,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )

    def ex_delete(self, group):
        """ 删除按钮 """
        t = group.search_label(lb.LABEL_DL)
        if t is not None:
            group.elements[lb.ELEMENT_DL] = lb.ElementImg(
                name=lb.ELEMENT_DL,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )

    def ex_group(self, group):
        """ 分组图标 """
        t = group.search_label(lb.LABEL_GRP)
        if t is not None:
            group.elements[lb.ELEMENT_GP] = lb.ElementImg(
                name=lb.ELEMENT_GP,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )

    def ex_like(self, group):
        """ 点赞按钮 """
        t = group.search_label(lb.LABEL_LK)
        if t is not None:
            group.elements[lb.ELEMENT_LK] = lb.ElementImg(
                name=lb.ELEMENT_LK,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )

    def ex_comment(self, group):
        """ 评论按钮 """
        t = group.search_label(lb.LABEL_CMT)
        if t is not None:
            group.elements[lb.ELEMENT_CM] = lb.ElementImg(
                name=lb.ELEMENT_CM,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )

    def ex_logo(self, group):
        """ Logo """
        ele_logo = [
            (lb.ELEMENT_LHL, group.search_label(lb.LABEL_LHL)),
            (lb.ELEMENT_LGB, group.search_label(lb.LABEL_LGB)),
            (lb.ELEMENT_LWD1, group.search_label(lb.LABEL_LWD)),
            (lb.ELEMENT_LWD2, group.search_label(lb.LABEL_LWD2)),
            (lb.ELEMENT_LWD3, group.search_label(lb.LABEL_LWD3)),
        ]
        ele_logo = filter(lambda v: v[1] is not None, ele_logo)
        ele_logo = max(ele_logo, key=lambda v: v[1].confidence, default=None)
        if ele_logo is not None:
            e, t = ele_logo
            group.elements[e] = lb.ElementImg(
                name=e,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )

    def ex_like_comment(self, group):
        """ 点赞-评论图标 """
        t = group.search_label(lb.LABEL_LK_CMT)
        if t is not None:
            group.elements[lb.ELEMENT_LC] = lb.ElementImg(
                name=lb.ELEMENT_LC,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )

    def ex_portrait_nickname(self, group):
        """ 头像和昵称 """
        t = group.search_label(lb.LABEL_PT_NN)
        if t is not None:
            # 原图大小
            shape = self.yolo_model.get_img_shape()
            w0 = shape['width']
            h0 = shape['height']
            # 裁剪头像-昵称区域
            img_ori = self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            h, w = img_ori.shape[:2]
            img_ex = self.yolo_model.crop_img(t.cx, t.cy+int(0.5*t.height), t.width, t.height*2)
            h_ex, w_ex = img_ex.shape[:2]
            # 分割线
            si = int(1.05 * h_ex)
            # print('separate [2]')
            # si = cva.separate2(
            #     img_ex, 
            #     ws=int(0.2 * h_ex), 
            #     dir='hor', 
            #     reverse=False,
            #     img_wb=self.yolo_model.crop_img(t.cx, t.cy, 1, t.height)
            # )
            # if si is None:
            #     print('separate [1]')
            #     si = cva.separate(
            #         img_ex, 
            #         ws=int(0.2 * h_ex), 
            #         dir='hor', 
            #         reverse=False,
            #         img_wb=self.yolo_model.crop_img(t.cx, t.cy, 1, t.height)
            #     )
            # si = int(si * w_ex)
            # 计算头像和昵称位置
            w_pt = si
            h_pt = h * 2
            w_nn = w - si
            h_nn = h
            cx_pt = t.cx * w0 - 0.5 * (w - w_pt)
            cy_pt = t.cy * h0 + 0.5 * h
            cx_nn = t.cx * w0 + 0.5 * (w - w_nn)
            cy_nn = t.cy * h0
            # 保存头像和昵称
            ele_portrait = lb.ElementImg(
                name=lb.ELEMENT_PT,
                cx=cx_pt/w0,
                cy=cy_pt/h0,
                width=w_pt/w0,
                height=h_pt/h0,
                img=self.yolo_model.crop_img(
                    cx_pt/w0, cy_pt/h0,
                    w_pt/w0, h_pt/h0
                )
            )
            ele_nickname = lb.ElementImg(
                name=lb.ELEMENT_NN,
                cx=cx_nn/w0,
                cy=cy_nn/h0,
                width=w_nn/w0,
                height=h_nn/h0,
                img=None
            )
            group.elements[lb.ELEMENT_PT] = ele_portrait
            group.elements[lb.ELEMENT_NN] = lb.ElementTxt(
                text=cht_to_chs(self.get_text(ele_nickname)),
                **ele_nickname.asdict()
            )

    def ex_describe(self, group):
        """ 分享语 """
        t = group.search_label(lb.LABEL_PT_NN_DC)
        if t is not None:
            x1, y1, x2, y2 = t.x1, t.y1, t.x2, t.y2
            if group.elements[lb.ELEMENT_PT].exists:
                ele = group.elements[lb.ELEMENT_PT]
                x1 = max(x1, ele.cx + 0.5 * ele.width)
            if group.elements[lb.ELEMENT_NN].exists:
                ele = group.elements[lb.ELEMENT_NN]
                y1 = max(y1, ele.cy + 0.5 * ele.height)
            ele_describe = lb.ElementImg(
                name=lb.ELEMENT_DS,
                cx=0.5*(x1+x2),
                cy=0.5*(y1+y2),
                width=x2-x1,
                height=y2-y1,
                img=None
            )
            group.elements[lb.ELEMENT_DS] = lb.ElementTxt(
                text=cht_to_chs(self.get_text(ele_describe)),
                **ele_describe.asdict()
            )

    def ex_time(self, group):
        # if (ele_twodot is not None and ele_twodot.exists) \
        #    or (ele_delete is not None and ele_delete.exists) \
        # :
        time_temps = self.label_conf.get_time_template()
        if self.screen == 1:
            ele_twodot = group.elements.get(lb.ELEMENT_TD, None)
            ele_delete = group.elements.get(lb.ELEMENT_DL, None)
            # ele_group = group.elements.get(lb.ELEMENT_GP, None)
            x1 = y1 = 0
            x2 = y2 = 1
            if ele_twodot is not None and ele_twodot.exists:
                x2 = min(x2, ele_twodot.x1)
                y1 = max(y1, ele_twodot.y1)
                y2 = min(y2, ele_twodot.y2)
            if ele_delete is not None and ele_delete.exists:
                x2 = min(x2, ele_delete.x1)
                y1 = max(y1, ele_delete.y1)
                y2 = min(y2, ele_delete.y2)
            # if ele_group is not None and ele_group.exists:
            #     x2 = min(x2, ele_group.x1)
            ele_time = lb.ElementImg(
                name=lb.ELEMENT_TM,
                cx=0.5*(x1+x2),
                cy=0.5*(y1+y2),
                width=(x2-x1),
                height=(y2-y1),
                img=None
            )
            # 匹配时间模版
            txt = self.get_text_hor(ele_time)
            # print('parse time text: ', txt)
            for t in time_temps:
                pattern = re.compile(r'%s' % t['pattern'])
                match = pattern.search(txt)
                if match:
                    s, e = match.span()
                    loc = {'match': match, 'ts': self.time_anchor}
                    exec(t['format'], globals(), loc)
                    group.elements[lb.ELEMENT_TM] = lb.ElementTxt(
                        text=loc['timestamp'],
                        **ele_time.asdict()
                    )
                    # 匹配 app name
                    app_name = cht_to_chs(txt[e:])
                    app_name = app_name.replace("删除", '')
                    app_name = app_name.replace("删", '')
                    app_name = app_name.replace("除", '')
                    app_name = app_name.replace("delete", '')
                    app_name = app_name.strip()
                    if app_name:
                        group.elements[lb.ELEMENT_AN] = lb.ElementTxt(
                            name=lb.ELEMENT_AN,
                            cx=ele_time.cx,
                            cy=ele_time.cy,
                            width=ele_time.width,
                            height=ele_time.height,
                            text=app_name,
                        )
                    break
        elif self.screen == 2:
            ele_like = group.elements.get(lb.ELEMENT_LK, None)
            ele_cmt = group.elements.get(lb.ELEMENT_CM, None)
            ele_like_cmt = group.elements.get(lb.ELEMENT_LC, None)
            h = 0
            if ele_like is not None and ele_like.exists:
                h = max(h, ele_like.height)
            if ele_cmt is not None and ele_cmt.exists:
                h = max(h, ele_cmt.height)
            if ele_like_cmt is not None and ele_like_cmt.exists:
                h = max(h, ele_like_cmt.height)
            txt = self.get_text_all()
            pos = self.get_text_pos(0, len(txt))
            pos = list(filter(lambda v: v[1] < 0.5, pos))
            txt = txt[0:len(pos)]
            print('parse time text: ', txt)

            needle = 0
            matches = []
            while(needle < len(txt)):
                matched = False
                for t in time_temps:
                    pattern = re.compile(r'%s' % t['pattern'])
                    m = pattern.search(txt, pos=needle)
                    if m:
                        s, e = m.span()
                        matches.append((t, m))
                        needle = e
                        matched = True
                        break
                if not matched:
                    break
            match = None
            match = matches[1] if len(matches) > 1 else matches[0] if len(matches) > 0 else None
            if match is not None:
                t, m = match
                s, e = m.span()
                txt_pos = self.get_text_pos(s, e)
                loc = {'match': m, 'ts': self.time_anchor}
                exec(t['format'], globals(), loc)
                group.elements[lb.ELEMENT_TM] = lb.ElementTxt(
                    name=lb.ELEMENT_TM,
                    cx=0.5, cy=txt_pos[0][1],
                    width=1, height=h,
                    text=loc['timestamp']
                )

            # for t in time_temps:
            #     pattern = re.compile(r'%s' % t['pattern'])
            #     match = None
            #     needle = 0
            #     while needle < len(txt):
            #         m = pattern.search(txt, pos=needle)
            #         if m:
            #             s, e = m.span()
            #             p = self.get_text_pos(s, e)
            #             if abs(p[0][0] - 0.5) < 0.05 \
            #                and p[0][1] < 0.5 \
            #                and abs(p[len(p)-1][0] - 0.5) < 0.05 \
            #                and p[len(p)-1][1] < 0.5 \
            #             :
            #                 match = m
            #                 break
            #             needle += 1
            #         else:
            #             break
            #     if match:
            #         s, e = match.span()
            #         txt_pos = self.get_text_pos(s, e)
            #         loc = {'match': match, 'ts': self.time_anchor}
            #         exec(t['format'], globals(), loc)
            #         group.elements[lb.ELEMENT_TM] = lb.ElementTxt(
            #             name=lb.ELEMENT_TM,
            #             cx=0.5, cy=txt_pos[0][1],
            #             width=1, height=h,
            #             text=loc['timestamp']
            #         )
            #         break


    def ex_poster(self, group):
        """ 海报 """
        # 场景1, 取中间的一块作为海报
        # if (ele_portrait is not None and ele_portrait.exists) \
        #    or (ele_nickname is not None and ele_nickname.exists) \
        #    or (ele_describe is not None and ele_describe.exists) \
        #    or (ele_twodot is not None and ele_twodot.exists) \
        #    or (ele_delete is not None and ele_describe.exists) \
        # :
        if self.screen == 1:
            ele_portrait = group.elements.get(lb.ELEMENT_PT, None)
            ele_nickname = group.elements.get(lb.ELEMENT_NN, None)
            ele_describe = group.elements.get(lb.ELEMENT_DS, None)
            ele_twodot = group.elements.get(lb.ELEMENT_TD, None)
            ele_delete = group.elements.get(lb.ELEMENT_DL, None)
            r = group.get_rect()
            x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
            if ele_portrait is not None and ele_portrait.exists:
                x1 = max(x1, ele_portrait.x2)
            if ele_nickname is not None and ele_nickname.exists:
                y1 = max(y1, ele_nickname.y2)
            if ele_describe is not None and ele_describe.exists:
                y1 = max(y1, ele_describe.y2)
            if ele_twodot is not None and ele_twodot.exists:
                y2 = min(y2, ele_twodot.y1)
            if ele_delete is not None and ele_delete.exists:
                y2 = min(y2, ele_delete.y1)
            if x2 > x1 and y2 > y1:
                group.elements[lb.ELEMENT_PS] = lb.ElementImg(
                    name=lb.ELEMENT_PS,
                    cx=0.5*(x1+x2),
                    cy=0.5*(y1+y2),
                    width=(x2-x1),
                    height=(y2-y1),
                    img=self.yolo_model.crop_img(
                        0.5*(x1+x2), 0.5*(y1+y2),
                        x2-x1, y2-y1
                    )
                )
        # 场景2, 把整张图当作海报
        # elif (ele_like is not None and ele_like.exists) \
        #      or (ele_comment is not None and ele_comment.exists) \
        #      or (ele_like_comment is not None and ele_like_comment.exists) \
        # :
        elif self.screen == 2:
            ele_time = group.elements.get(lb.ELEMENT_TM, None)
            ele_like = group.elements.get(lb.ELEMENT_LK, None)
            ele_cmt = group.elements.get(lb.ELEMENT_CM, None)
            ele_like_cmt = group.elements.get(lb.ELEMENT_LC, None)
            x1 = y1 = 0
            x2 = y2 = 1
            if ele_time is not None and ele_time.exists:
                y1 = max(y1, ele_time.cy + 0.5 * ele_time.height)
            if ele_like is not None and ele_like.exists:
                y2 = min(y2, ele_like.cy - 0.5 * ele_like.height)
            if ele_cmt is not None and ele_cmt.exists:
                y2 = min(y2, ele_cmt.cy - 0.5 * ele_cmt.height)
            if ele_like_cmt is not None and ele_like_cmt.exists:
                y2 = min(y2, ele_like_cmt.cy - 0.5 * ele_like_cmt.height)
            group.elements[lb.ELEMENT_PS] = lb.ElementImg(
                name=lb.ELEMENT_PS,
                cx=0.5 * (x1 + x2),
                cy=0.5 * (y1 + y2),
                width=(x2 - x1),
                height=(y2 - y1),
                img=self.yolo_model.crop_img(
                    0.5 * (x1 + x2),
                    0.5 * (y1 + y2),
                    x2 - x1,
                    y2 - y1
                )
            )

    def ex_like_comment_num(self, group):
        """ 点赞数/评论数 """
        n_like = 0
        n_comment = 0
        bottom = group.bottom
        lc = None
        # 场景2
        if self.screen == 2:
            t = group.search_label(lb.LABEL_LK)
            if t is None:
                t = group.search_label(lb.LABEL_CMT)
            if t is None:
                t = group.search_label(lb.LABEL_LK_CMT)
            if t is not None:
                r = lb.Rect(0.75, t.cy, 0.5, t.height)
                img = self.yolo_model.crop_img(r.cx, r.cy, r.width, r.height)
                # 点赞/评论图标检测
                lt = None
                ct = None
                for _t in self.tuples:
                    if _t.label == lb.LABEL_LS and r.inside(_t.cx, _t.cy):
                        if lt is None:
                            lt = _t
                        elif _t.confidence > lt.confidence:
                            lt = _t
                    if _t.label == lb.LABEL_CS and r.inside(_t.cx, _t.cy):
                        if ct is None:
                            ct = _t
                        elif _t.confidence > ct.confidence:
                            ct = _t
                if lt is not None and ct is not None:
                    s = self.yolo_model.get_img_shape()
                    h0, w0 = img.shape[:2]
                    # 局部坐标
                    m1 = np.array([
                        [r.width, 0, 0],
                        [0, r.height, 0],
                        [r.x1, r.y1, 1]
                    ])
                    m2 = np.linalg.inv(m1)
                    lr = lb.Rect(
                        w0 * (lt.cx * m2[0][0] + lt.cy * m2[1][0] + m2[2][0]),
                        h0 * (lt.cx * m2[0][1] + lt.cy * m2[1][1] + m2[2][1]),
                        lt.width * s['width'],
                        lt.height * s['height']
                    )
                    cr = lb.Rect(
                        w0 * (ct.cx * m2[0][0] + ct.cy * m2[1][0] + m2[2][0]),
                        h0 * (ct.cx * m2[0][1] + ct.cy * m2[1][1] + m2[2][1]),
                        ct.width * s['width'],
                        ct.height * s['height']
                    )
                    # cv2.rectangle(img, (int(lr.x1), int(lr.y1)), (int(lr.x2), int(lr.y2)), (0,255,0), 2)
                    # cv2.rectangle(img, (int(cr.x1), int(cr.y1)), (int(cr.x2), int(cr.y2)), (0,255,0), 2)
                    # cva.show_img('like-comment', img)
                    # cva.wait_img()
                    # 点赞数
                    if int(lr.x2) >= int(cr.x1):
                        return
                    img_cp = img[:, int(lr.x2):int(cr.x1), :]
                    img_cp = cva.expand(img_cp, ratio=(2, 2), fill=img[5,5,])
                    img_cp = cv2.resize(img_cp, (img_cp.shape[1] * 2, img_cp.shape[0] * 2))
                    ret = es_extract_text(self.checker_id, img_cp, ch=False)
                    print('like es_extract_text: ', ret)
                    s = ''.join(map(lambda v: v['text'], ret))
                    if s == '7' and ret[0]['confidence'] < 0.95:
                        s = '1'
                    if s.isdigit():
                        n_like = int(s)
                    # 评论数
                    img_cp = img[:, int(cr.x2):, :]
                    img_cp = cva.expand(img_cp, ratio=(2, 2), fill=img[5,5,])
                    img_cp = cv2.resize(img_cp, (img_cp.shape[1] * 2, img_cp.shape[0] * 2))
                    # cva.show_img('xxxx', img_cp)
                    # cva.wait_img()
                    ret = es_extract_text(self.checker_id, img_cp, ch=False)
                    print('comment es_extract_text: ', ret)
                    s = ''.join(map(lambda v: v['text'], ret))
                    if s == '7' and ret[0]['confidence'] < 0.95:
                        s = '1'
                    if s.isdigit():
                        n_comment = int(s)
        # 场景1
        elif self.screen == 1 and bottom is not None:
            lk_tuples = []
            cm_tuples = []
            for t in self.tuples:
                if bottom.inside(t.cx, t.cy):
                    if t.label == lb.LABEL_LP:
                        lk_tuples.append(t)
                    elif t.label == lb.LABEL_CP:
                        cm_tuples.append(t)
            lk_tuples = sorted(lk_tuples, key=lambda v: v.confidence, reverse=True)
            cm_tuples = sorted(cm_tuples, key=lambda v: v.confidence, reverse=True)
            lk_rects = []
            cm_rects = []
            lk_img = None
            cm_img = None
            # 详情页
            if len(lk_tuples) > 0 or len(cm_tuples) > 0:
                if len(lk_tuples) > 0:
                    t = lk_tuples[0]
                    lk_img = self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
                    h0, w0 = lk_img.shape[:2]
                    # 放大图片
                    lk_img = cv2.resize(lk_img, (w0*2, h0*2))
                    # 搜索头像
                    lk_rects = cva.search_rect(lk_img, space=50, fill=(5,5), show=False)
                    # 过滤
                    lk_rects = cva.search_rect_filter(lk_rects)
                    # 评论数
                    n_like = len(lk_rects)
                if len(cm_tuples) > 0:
                    t = cm_tuples[0]
                    cm_img = self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
                    h0, w0 = cm_img.shape[:2]
                    # 放大图片
                    cm_img = cv2.resize(cm_img, (w0*2, h0*2))
                    # 搜索头像
                    cm_rects = cva.search_rect(cm_img, space=50, fill=(5,5), show=False)
                    # 过滤
                    cm_rects = cva.search_rect_filter(cm_rects)
                    # 点赞数
                    n_comment = len(cm_rects)
                # 显示头像检测结果(测试用)
                # if lk_img is not None or cm_img is not None:
                #     if lk_img is not None:
                #         for x, y, w, h in lk_rects:
                #             cv2.rectangle(lk_img, (x, y), (x+w, y+h), (0,255,0), 2)
                #         cva.show_img('like', lk_img)
                #     if cm_img is not None:
                #         for x, y, w, h in cm_rects:
                #             cv2.rectangle(cm_img, (x, y), (x+w, y+h), (0,255,0), 2)
                #         cva.show_img('comment', cm_img)
                #     cva.wait_img()
            # 信息流页
            else:
                # 该区域的所有文字
                words_lc = list(filter(
                    lambda w: bottom.inside(w['cx'], w['cy']), self.words
                ))
                if len(words_lc) > 0:
                    s = ''.join(map(lambda w: w['word'], words_lc))
                    print('paddle text: ', s)
                    has_lc = True
                    # 没有评论
                    if ':' not in s and '：' not in s:
                        # 没有点赞图标
                        v = False
                        for t in self.tuples:
                            if t.label == lb.LABEL_LS and bottom.inside(t.cx, t.cy):
                                v = True
                                break
                        has_lc = v
                    if has_lc:
                        lc = s
                        # 评论数
                        l1 = re.split(r':|：', s)
                        n_comment = len(l1) - 1
                        # 点赞数
                        n_like = 0
                        l2 = re.split(r',|，', l1[0])
                        n_like = len(l2) - 1 if n_comment > 0 else len(l2)
        group.elements[lb.ELEMENT_NL] = lb.ElementTxt(
            name=lb.ELEMENT_NL,
            cx=0, cy=0, width=0, height=0,
            text=n_like
        )
        group.elements[lb.ELEMENT_NC] = lb.ElementTxt(
            name=lb.ELEMENT_NC,
            cx=0, cy=0, width=0, height=0,
            text=n_comment
        )
        print('bottom: ', bottom)
        print('like-comment words: ', lc)
        print('n_like[%d],  n_comment[%d]' % (n_like, n_comment))

    def ex_mpcode(self, group):
        """ 小程序码 """
        # 豌豆
        t = group.search_label(lb.LABEL_MWD)
        if t is not None:
            # 小程序码
            group.elements[lb.ELEMENT_MWD] = lb.ElementImg(
                name=lb.ELEMENT_MWD,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )
            # 定位小程序码上方的识别码
            w = t.width
            h = 0.35 * t.height
            x1, x2 = t.x1, t.x2
            y1, y2 = t.y1 - h, t.y1 + h
            h = 2 * h
            cx = 0.5 * (x1 + x2) 
            cy = 0.5 * (y1 + y2)
            img = self.yolo_model.crop_img(cx, cy, w, h).copy()
            group.elements[lb.ELEMENT_RWDI] = lb.ElementImg(
                name=lb.ELEMENT_RWDI,
                cx=cx, cy=cy,
                width=w, height=h,
                img=img
            )
            # OCR 检测
            # ret = es_extract_text(self.checker_id, img, ch=False)
            ret = pd_extract_text(img)
            # ret = pd_extract_text_mini(img)
            if len(ret) > 0:
                s = ''.join(map(lambda v: v['text'], ret[:1]))
                group.elements[lb.ELEMENT_RWD] = lb.ElementTxt(
                    name=lb.ELEMENT_RWD,
                    cx=cx, cy=cy,
                    width=w, height=h,
                    text=s
                )

    def ex_qrcode(self, group):
        """ 二维码 """
        def redirect_url(url):
            """ 重定向 """
            try:
                res = urlopen(url, timeout=2)
                return res.geturl()
            except URLError as e:
                print('request error [%s]: %s' % (url, e), file=sys.stderr)
            return None
        # 二维码
        t = group.search_label(lb.LABEL_QR)
        if t is not None:
            group.elements[lb.ELEMENT_QR] = lb.ElementImg(
                name=lb.ELEMENT_QR,
                cx=t.cx, cy=t.cy,
                width=t.width, height=t.height,
                img=self.yolo_model.crop_img(t.cx, t.cy, t.width, t.height)
            )
        # 二维码内容解析
        if group.elements[lb.ELEMENT_QR].exists:
            e = group.elements[lb.ELEMENT_QR]
            img = self.yolo_model.crop_img(e.cx, e.cy, 1.5 * e.width, 1.5 * e.height)
            res, points = self.qr_parser.detectAndDecode(img)
            if len(res) > 0:
                url = res[0]
                url_rd = redirect_url(url)
                if url_rd is not None:
                    url = url_rd
                group.elements[lb.ELEMENT_QRC] = lb.ElementTxt(
                    name=lb.ELEMENT_QRC,
                    cx=t.cx, cy=t.cy,
                    width=t.width, height=t.height,
                    text=url
                )

    def check(self, img_source, ts=None):
        ret = {'screen': -1, 'blocks': []}
        # 匹配模版
        self.match(img_source, ts)
        ret['screen'] = self.screen
        if ret['screen'] < 0:
            return ret

        # 解析元素
        for idx, g in enumerate(self.matched_groups):
            self.ex_portrait_nickname(g)
            self.ex_describe(g)
            self.ex_twodot(g)
            self.ex_delete(g)
            self.ex_group(g)
            self.ex_like(g)
            self.ex_comment(g)
            self.ex_like_comment(g)
            self.ex_logo(g)
            self.ex_time(g)
            self.ex_poster(g)
            self.ex_like_comment_num(g)
            self.ex_mpcode(g)
            self.ex_qrcode(g)
        
        # 构造返回值
        for g in self.matched_groups:
            block = {}
            for ele in g.elements.values():
                if lb.in_filter(ele):
                    block[ele.xname] = ele.dumps()
            ret['blocks'].append(block)
        return ret

    def quit(self):
        # self.qr_pool.quit()
        pass


class SiftDetector(namedtuple('SiftDetector', 'img_path width height key_points descriptor')):
    @property
    def p(self):
        return self.img_path

    @property
    def w(self):
        return self.width

    @property
    def h(self):
        return self.height

    @property
    def kps(self):
        return self.key_points

    @property
    def dsc(self):
        return self.descriptor


class ImageMatcher(object):
    def __init__(self, match_id):
        self.match_id = match_id
        # 初始化 sift
        self.sift = cv2.SIFT_create()
        # 初始化匹配方式(最临近匹配, 一种近似匹配法, 不追求完美, 但速度快)
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # sift 缓存
        self.sift_buffer = {}

    def _is_rectangle(self, p1, p2, p3, p4):
        """
        是否是矩形
        """
        def dist(_p1, _p2):
            return (_p1[0] - _p2[0])**2 + (_p1[1] - _p2[1])**2
        x_c = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
        y_c = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0
        d1 = dist(p1, (x_c, y_c))
        d2 = dist(p2, (x_c, y_c))
        d3 = dist(p3, (x_c, y_c))
        d4 = dist(p4, (x_c, y_c))
        d_max = np.max([d1, d2, d3, d4])
        d_min = np.min([d1, d2, d3, d4])
        return (d_max - d_min) < 0.05 * d_min

    def _detect(self, img_path, half=False, save=True):
        if img_path in self.sift_buffer:
            return self.sift_buffer[img_path]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if half:
            h = int(0.5 * h)
            img = img[:h, :, :]
        k, d = self.sift.detectAndCompute(img, None)
        detector = SiftDetector(
            img_path=img_path,
            width=w,
            height=h,
            key_points=k,
            descriptor=d
        )
        if save:
            self.sift_buffer[img_path] = detector
        return detector

    def _match(self, detector1, detector2, show=False):
        d1, d2 = detector1, detector2
        print('matching %s vs %s' % (d1.p, d2.p))
        matches = self.flann.knnMatch(d1.dsc, d2.dsc, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        # 已经匹配到足够多的特征点
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([d1.kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([d2.kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 计算多个二维点对之间的最优单映射变换矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # 透视变换, 定位匹配目标
            h, w = d1.h, d1.w
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)
            dst_sqz = np.squeeze(dst)
            # 形状判断, 如果不是矩形则认为该模版匹配失败
            if self._is_rectangle(dst_sqz[0], dst_sqz[1], dst_sqz[2], dst_sqz[3]):
                print('matched %s vs %s' % (d1.p, d2.p))
                if show:
                    img1 = cv2.imread(d1.p)
                    img2 = cv2.imread(d2.p)
                    matches_mask = mask.ravel().tolist()
                    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)
                    draw_params = dict(
                        matchColor=(0, 255, 0),
                        singlePointColor=None,
                        matchesMask=matches_mask,
                        flags=0
                    )
                    img3 = cv2.drawMatches(img1, d1.kps, img2, d2.kps, good, None, **draw_params)
                    # cva.show_img('match', img3)
                    # cva.wait_img()
                return len(good)
        print('match failed %s vs %s' % (d1.p, d2.p))
        return 0

    def match(self, img_path_match, img_path_temp, half=False, show=False):
        # 保存和二次加载匹配图片
        img2 = cv2.imread(img_path_match)
        tmp_file = '%s/match_img_%02d.jpg' % (DIR_TMP, self.match_id)
        cv2.imwrite(tmp_file, img2)
        img2 = cv2.imread(tmp_file)

        # 检测特征点
        d1 = self._detect(img_path_temp, half=half, save=True)
        d2 = self._detect(img_path_match, half=False, save=False)

        # 匹配特征点
        return self._match(d1, d2, show=show)


if __name__ == '__main__':
    from src.element_check.down import get_one_image, download_image

    # img_path_temp = '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_07.jpeg'
    # img_path_match = '/Users/penghuan/Documents/liuyi/gubi-wechat/gubi-wechat-test/610087dc4bc7d867920e7b52.jpeg'
    # im = ImageMatcher(99)
    # n_match = im.match(img_path_match, img_path_temp, half=True, show=False)
    # print(n_match)

    # img_path = '/Users/penghuan/Documents/liuyi/gubi-wechat/gubi-wechat-test/610090af4bc7d867920e7b63.jpeg'
    # img_path = '/Users/penghuan/Documents/liuyi/like-comment/gubi/7a0e731e50acbdd62dc098908db36ea0.jpg'
    # img_path = '/Users/penghuan/Documents/liuyi/like-comment/hualala/4b409c1ac97f941d37f220f9719a717d.png'
    # img_path = '/Users/penghuan/Documents/liuyi/gubi-wechat/gubi-wechat-test/90882b4d4f7c7e71542b0e7c393404a8.jpg'
    # img_path = '/Users/penghuan/Desktop/111.png'

    # url = 'http://10.200.11.244:8000/wandou-wechat/%E8%B1%8C%E8%B1%86%E6%80%9D%E7%BB%B4%2B%E8%B1%8C%E8%B1%86%E7%9B%8A%E6%99%BA/inte_list_logo/'
    # name_regex = r'c5b31'
    # img_name = get_one_image(url, name_regex)
    # if not img_name:
    #     raise Exception('xxoo')
    # img_source = '%s/%s' % (url, img_name)
    img_source = 'https://hll-cdn.oss-accelerate.aliyuncs.com/prod/hll-activity/2beda47f0be4c47dbfbf85d07f3fce41'

    checker = LabelChecker(99)

    # ret = checker.check(img_path, int(time.time()))
    # print(list(map(
    #     lambda v: v.keys(), ret['blocks']
    # )))

    checker.match(img_source, debug=True)
    print('screen: ', checker.screen)
    img_showed = False
    for idx, g in enumerate(checker.matched_groups):
        checker.ex_portrait_nickname(g)
        checker.ex_describe(g)
        checker.ex_twodot(g)
        checker.ex_delete(g)
        checker.ex_group(g)
        checker.ex_like(g)
        checker.ex_comment(g)
        checker.ex_like_comment(g)
        checker.ex_logo(g)
        checker.ex_time(g)
        checker.ex_poster(g)
        checker.ex_like_comment_num(g)
        checker.ex_mpcode(g)
        checker.ex_qrcode(g)

        labels = list(map(
            lambda v1: v1.name, filter(
                lambda v2: v2.exists, g.elements.values()
            )
        ))
        print('checked labels: ', labels)

        # cva.show_img('pt_%d' % idx, checker.crop_img(g.elements[lb.ELEMENT_PT]))
        # cva.show_img('nn_%d' % idx, checker.crop_img(g.elements[lb.ELEMENT_NN]))
        # cva.show_img('ds_%d' % idx, checker.crop_img(g.elements[lb.ELEMENT_DS]))
        # if g.elements[lb.ELEMENT_PS].exists:
        #     img_showed = True
        #     cva.show_img('ps_%d' % idx, checker.crop_img(g.elements[lb.ELEMENT_PS]))
        # if g.elements[lb.ELEMENT_LHL].exists:
        #     img_showed = True
        #     cva.show_img('logo_%d' % idx, checker.crop_img(g.elements[lb.ELEMENT_LHL]))
        # if g.elements[lb.ELEMENT_LGB].exists:
        #     img_showed = True
        #     cva.show_img('logo_%d' % idx, checker.crop_img(g.elements[lb.ELEMENT_LGB]))

        print('nickname: ', g.elements[lb.ELEMENT_NN].text if g.elements[lb.ELEMENT_NN].exists else None)
        print('describe: ', g.elements[lb.ELEMENT_DS].text if g.elements[lb.ELEMENT_DS].exists else None)
        print('app name: ', g.elements[lb.ELEMENT_AN].text if g.elements[lb.ELEMENT_AN].exists else None)
        ts = g.elements[lb.ELEMENT_TM].text if g.elements[lb.ELEMENT_TM].exists else None
        print('time    : ', time.strftime('%Y%m%d %H:%M', time.localtime(ts)) if ts is not None else None)
        print('like    : ', g.elements[lb.ELEMENT_NL].text if g.elements[lb.ELEMENT_NL].exists else None)
        print('comment : ', g.elements[lb.ELEMENT_NC].text if g.elements[lb.ELEMENT_NC].exists else None)
        print('recode  : ', g.elements[lb.ELEMENT_RWD].text if g.elements[lb.ELEMENT_RWD].exists else None)
        print('qrcode  : ', g.elements[lb.ELEMENT_QRC].text if g.elements[lb.ELEMENT_QRC].exists else None)
        logo = list(filter(lambda v: g.elements[v].exists, [lb.ELEMENT_LHL, lb.ELEMENT_LGB, lb.ELEMENT_LWD1, lb.ELEMENT_LWD2, lb.ELEMENT_LWD3, lb.ELEMENT_LWD4]))
        print('logo    : ', logo[0] if len(logo) > 0 else None)
        print()
    if img_showed:
        cva.wait_img()

    checker.quit()
