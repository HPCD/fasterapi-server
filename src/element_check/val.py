# coding: utf-8

import os
import sys
import shutil
import re
import csv
import time
import random
import numpy as np
import cv2
from pathlib import Path
from collections import namedtuple
from PIL import ImageFont, ImageDraw, Image
from down import get_all_image
import src.element_check.cv_app as cva
from src.element_check.extract import init_extract_worker, stop_extract_worker, extract_element
from src.element_check.match import init_match_worker, stop_match_worker, _match_poster
import src.element_check.label as lb


DIR_VAL_EXT = os.path.realpath(os.path.join('val', 'extract'))


class ImgText:
    font = ImageFont.truetype("/Library/Fonts/Songti.ttc", 21)
    def __init__(self, text, width):
        # 预设宽度 可以修改成你需要的图片宽度
        self.width = width
        # 文本
        self.text = text
        # 段落 , 行数, 行高
        self.duanluo, self.note_height, self.line_height = self.split_text()

    def get_duanluo(self, text):
        txt = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt)
        # 所有文字的段落
        duanluo = ""
        # 宽度总和
        sum_width = 0
        # 几行
        line_count = 1
        # 行高
        line_height = 0
        for char in text:
            width, height = draw.textsize(char, ImgText.font)
            sum_width += width
            if sum_width > self.width: # 超过预设宽度就修改段落 以及当前行数
                line_count += 1
                sum_width = 0
                duanluo += '\n'
            duanluo += char
            line_height = max(height, line_height)
        if not duanluo.endswith('\n'):
            duanluo += '\n'
        return duanluo, line_height, line_count

    def split_text(self):
        """ 按规定宽度分组 """
        max_line_height, total_lines = 0, 0
        allText = []
        for text in self.text.split('\n'):
            duanluo, line_height, line_count = self.get_duanluo(text)
            max_line_height = max(line_height, max_line_height)
            total_lines += line_count
            allText.append((duanluo, line_count))
        line_height = max_line_height
        total_height = total_lines * line_height
        return allText, total_height, line_height

    def draw_text(self, from_path, to_path, x=0, y=0):
        """ 绘图以及文字 """
        # note_img = Image.open(from_path).convert("RGBA")
        note_img = Image.open(from_path)
        draw = ImageDraw.Draw(note_img)
        for duanluo, line_count in self.duanluo:
            draw.text((x, y), duanluo, fill=(0, 0, 0), font=ImgText.font)
            y += self.line_height * line_count
        note_img.save(to_path)


def read_csv(csv_path, *headers, delimiter=','):
    p = Path(csv_path)
    assert p.is_file()
    Row = namedtuple('Row', headers)
    with open(str(p), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)
        for row in reader:
            row = Row(*row)
            yield row


def val_extract_one(img_path, val_path='val/test.jpg'):
    img_ori = cv2.imread(img_path)
    h0, w0 = img_ori.shape[:2]
    ret = extract_element(img_path, int(time.time()))
    if ret is not None:
        blocks = []
        for block in ret['blocks']:
            imgs = []
            txts = []
            # 头像
            portrait = block['user_img']['content']
            if portrait is not None:
                imgs.append(cva.decode_image(portrait))
            # 昵称
            nickname = block['user_name']['content']
            if nickname is not None:
                txts.append(nickname)
            # 分享语
            describe = block['comment']['content']
            if describe is not None:
                txts.append(describe)
            # 海报
            poster = block['share_pictures']['content']
            if poster is not None:
                imgs.append(cva.decode_image(poster))
            # 分组图标
            group = block['grouping_icon']['content']
            if group is not None:
                imgs.append(cva.decode_image(group))
            # 时间
            ts = block['publication_time']['content']
            if ts is not None:
                txts.append(time.strftime('%Y%m%d %H:%M', time.localtime(ts)))
            # 删除按钮
            delete = block['delete_button']['content']
            if delete is not None:
                imgs.append(cva.decode_image(delete))
            # 两点按钮
            twodot = block['two_point_icon']['content']
            if twodot is not None:
                imgs.append(cva.decode_image(twodot))
            # 点赞按钮
            like = block['like_icon']['content']
            if like is not None:
                imgs.append(cva.decode_image(like))
            # 评论按钮
            comment = block['comment_icon']['content']
            if comment is not None:
                imgs.append(cva.decode_image(comment))
            # 点赞/评论图标
            like_comment = block['like_comment_icon']['content']
            if like_comment is not None:
                imgs.append(cva.decode_image(like_comment))
            # Logo
            ele_logo = [
                ("画啦啦", block['logo_hualala']['content']),
                ("咕比启蒙", block['logo_gubi']['content']),
                ("豌豆思维", block['logo_wandou_siwei']['content']),
                ("豌豆口才", block['logo_wandou_koucai']['content']),
                ("豌豆益智", block['logo_wandou_yizhi']['content']),
            ]
            ele_logo = list(filter(lambda v: v[1] is not None, ele_logo))
            if len(ele_logo) > 0:
                logo_name, logo_img = ele_logo[0]
                imgs.append(cva.decode_image(logo_img))
                txts.append('Logo: %s' % logo_name)
            # APP 名称
            app = block['app_name']['content']
            if app is not None:
                txts.append(app)
            # 点赞/评论数
            n_like = block['n_like']['content']
            n_comment = block['n_comment']['content']
            if n_like is not None and n_comment is not None:
                txts.append('点赞: %d, 评论: %d' % (n_like, n_comment))
            # 小程序码
            mpcode_wandou = block['mpcode_wandou']['content']
            if mpcode_wandou is not None:
                imgs.append(cva.decode_image(mpcode_wandou))
            # 海报识别码
            recode_wandou = block['recode_wandou']['content']
            if recode_wandou is not None:
                txts.append('海报识别码: %s' % recode_wandou)
            recode_wandou_img = block['recode_wandou_img']['content']
            if recode_wandou_img is not None:
                imgs.append(cva.decode_image(recode_wandou_img))
            # 二维码
            qrcode = block['qrcode']['content']
            if qrcode is not None:
                imgs.append(cva.decode_image(qrcode))
            qrcode_content = block['qrcode_content']['content']
            if qrcode_content is not None:
                txts.append("二维码解析成功")
            blocks.append((imgs, txts))

        # 整合图片
        x_space = 20
        y_space = 20
        w = 3 * w0 + 2 * x_space
        h = 0
        for imgs, txts in blocks:
            for img in imgs:
                _h, _w = img.shape[:2]
                h = h + _h + y_space
            h = h + y_space
        h = max(h, h0)
        img_new = np.zeros((h, w, 3), np.uint8)
        img_new[:,:,:] = 175
        img_new[0:h0, 0:w0, :] = img_ori
        w_needle = w0 + x_space
        h_needle = 0
        txt_buffer = []
        for imgs, txts in blocks:
            txt_buffer.append(('\n\n'.join(txts), h_needle))
            for i, img in enumerate(imgs):
                _h, _w = img.shape[:2]
                img_new[h_needle:h_needle+_h, w_needle:w_needle+_w, :] = img
                h_needle = h_needle + _h + y_space
            img_new[h_needle:h_needle+y_space, w_needle:, :] = 50
            h_needle = h_needle + y_space
        cv2.imwrite(val_path, img_new)

        w_needle = 2 * (w0 + x_space)
        for txt, h_needle in txt_buffer:
            ImgText(txt, int(0.9 * w0)).draw_text(
                val_path, 
                val_path,
                w_needle, h_needle
            )


def val_match_one(img_path, poster_paths, val_path='val/test.jpg'):
    match_path = 'img/val_match.jpg'
    matched_temp_path = None
    img_ori = cv2.imread(img_path)
    h0, w0 = img_ori.shape[:2]
    ret = extract_element(img_path, int(time.time()))
    if ret is not None:
        for block in ret['blocks']:
            poster = block['share_pictures']['content']
            if poster is not None:
                img_poster = cva.decode_image(poster)
                cv2.imwrite(match_path, img_poster)
                for pp in poster_paths:
                    if _match_poster(match_path, [pp], []):
                        matched_temp_path = pp
                        break
            if not matched_temp_path:
                break
    # 整合图片
    img1 = cv2.imread(img_path)
    if matched_temp_path:
        img2 = cv2.imread(matched_temp_path)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        x_space = 20
        w = w1 + w2 + x_space
        h = max(h1, h2)
        img_new = np.zeros((h, w, 3), np.uint8)
        img_new[:,:,:] = 175
        img_new[0:h1, 0:w1, :] = img1
        img_new[0:h2, w1+x_space:w, :] = img2
        cv2.imwrite(val_path, img_new)
    else:
        cv2.imwrite(val_path, img1)


def val_extract(img_paths, val_dir, prefix):
    for idx, p in enumerate(img_paths):
        fname, ftype = p.parts[-1].split('.')
        fname = '%s_%s' % (prefix, fname)
        print('validation: ', idx, fname, ftype)
        val_extract_one(str(p), os.path.join(val_dir, "%s.jpg" % fname))


def sample_img(img_dir, img_num):
    """ 随机抽选图片 """
    p = Path(img_dir)
    pattern = re.compile(r".*\.(jpg|png|jpeg)$")
    all_img_paths = []
    for p in p.glob("**/*"):
        match = re.search(pattern, str(p))
        if match:
            all_img_paths.append(p)
    return random.sample(all_img_paths, min(img_num, len(all_img_paths)))



if __name__ == '__main__':
    # from src.utils import download_img
    from src.element_check.down import get_one_image, download_image

    # 初始化
    # import logging
    # logging.basicConfig(level=logging.INFO)
    lb.ELE_FILTER = lb.ELE_FILTER_ALL
    init_extract_worker()
    init_match_worker()

    # 验证一张图(extract)
    # img_url = 'https://hualala-common.oss-cn-shenzhen.aliyuncs.com/prod/hll-activity/614fef3770b5353f655cdfde.jpg'
    # img_path = './val/origin.jpg'
    # download_img(img_url, img_path)
    # img_path = '/Users/penghuan/Documents/liuyi/like-comment/hualala/3f8ac6a54ee50e19e392ebd0b0f0f5d9.png'
    # img_path = '/Users/penghuan/Documents/liuyi/gubi-wechat/gubi-wechat-test/610087dc4bc7d867920e7b52.jpeg'
    # img_path = '/Users/penghuan/Documents/liuyi/like-comment/hualala/5f163d78321115d6a35fc0c7c738cb29.png'
    # img_path = '/Users/penghuan/Desktop/111.png'
    # val_extract_one(img_path)

    url = 'http://10.200.11.244:8000/hualala-wechat/images/success/'
    name_regex = r'ab5a4'
    img_name = get_one_image(url, name_regex)
    if not img_name:
        raise Exception('xxoo')
    name, fmt = img_name.split('.')
    img_path = os.path.join('img', 'tmp.%s' % fmt)
    download_image(url, img_name, img_path)
    val_extract_one(img_path)


    # 批量验证(extract)
    # train_dir = '/Users/penghuan/Documents/liuyi/like-comment/hualala'
    # val_dir = './val/extract/hualala'
    # prefix = 'hualala'
    # for p in [Path(os.path.realpath(val_dir))]:
    #     if p.exists():
    #         shutil.rmtree(str(p))
    #     os.mkdir(str(p))
    # val_extract(sample_img(train_dir, 250), val_dir, prefix)

    # val_dir = './val/extract/wandou'
    # p = Path(os.path.realpath(val_dir))
    # if p.exists():
    #     shutil.rmtree(str(p))
    # os.mkdir(str(p))
    # url = 'http://10.200.11.244:8000/wandou-wechat/%E8%B1%8C%E8%B1%86%E6%80%9D%E7%BB%B4%2B%E8%B1%8C%E8%B1%86%E7%9B%8A%E6%99%BA/inte_list_logo/'
    # cnt = 0
    # for img_name in get_all_image(url):
    #     print(img_name)
    #     name, fmt = img_name.split('.')
    #     img_path = os.path.join('img', 'tmp.%s' % fmt)
    #     val_path = os.path.join(val_dir, '%s.jpg' % name)
    #     download_image(url, img_name, img_path)
    #     val_extract_one(img_path, val_path=val_path)
    #     cnt += 1
    #     if cnt >= 100:
    #         break
        


    # 验证一张图(match)
    # poster_dir = '/Users/penghuan/Documents/liuyi/gubi-wechat/poster'
    # img_path = '/Users/penghuan/Documents/liuyi/gubi-wechat/gubi-wechat-test/610087dc4bc7d867920e7b52.jpeg'
    # poster_paths = list(map(
    #     lambda v: str(v), sample_img(poster_dir, sys.maxsize)
    # ))
    # val_match_one(img_path, poster_paths)

    stop_extract_worker()
    stop_match_worker()

