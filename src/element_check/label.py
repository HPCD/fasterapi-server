# coding: utf-8

import sys
import numpy as np
from itertools import chain
from collections import namedtuple
from src.element_check.cv_app import encode_image


LABEL_PT_NN = 'portrait-nickname'
LABEL_PT_NN_DC = 'portrait-nickname-describe'
LABEL_DL = 'delete'
LABEL_TD = 'twodot'
LABEL_LK = 'like'
LABEL_CMT = 'comment'
LABEL_LK_CMT = 'like-comment'
LABEL_GRP = 'group'
LABEL_LHL = 'logo-hualala'
LABEL_LGB = 'logo-gubi'
LABEL_LP = 'like-portrait'
LABEL_CP = 'comment-portrait'
LABEL_LS = 'like-single'
LABEL_CS = 'comment-single'
LABEL_LWD = 'logo-wandou'
LABEL_MWD = 'mpcode-wandou'
LABEL_QR = 'qrcode'
LABEL_LWD2 = 'logo-wandou-koucai'
LABEL_LWD3 = 'logo-wandou-yizhi'

ELEMENT_PT = 'portrait'
ELEMENT_NN = 'nickname'
ELEMENT_DS = 'describe'
ELEMENT_PS = 'poster'
ELEMENT_GP = 'group'
ELEMENT_TM = 'time'
ELEMENT_DL = 'delete'
ELEMENT_TD = 'twodot'
ELEMENT_LK = 'like'
ELEMENT_CM = 'comment'
ELEMENT_LC = 'like-comment'
ELEMENT_AN = 'app_name'
ELEMENT_LHL = 'logo_hualala'
ELEMENT_LGB = 'logo_gubi'
ELEMENT_NL = 'n_like'
ELEMENT_NC = 'n_comment'
ELEMENT_LWD = 'logo_wandou'
ELEMENT_MWD = 'mpcode_wandou'
ELEMENT_RWD = 'recode_wandou'
ELEMENT_RWDI = 'recode_wandou_img'
ELEMENT_QR = 'qrcode'
ELEMENT_QRC = 'qrcode_content'
ELEMENT_LWD1 = 'logo_wandou_siwei'
ELEMENT_LWD2 = 'logo_wandou_koucai'
ELEMENT_LWD3 = 'logo_wandou_yizhi'
ELEMENT_LWD4 = 'logo_wandou_suzhi'

ELE_NAME = {
    ELEMENT_PT: 'user_img',
    ELEMENT_NN: 'user_name',
    ELEMENT_DS: 'comment',
    ELEMENT_PS: 'share_pictures',
    ELEMENT_GP: 'grouping_icon',
    ELEMENT_TM: 'publication_time',
    ELEMENT_DL: 'delete_button',
    ELEMENT_TD: 'two_point_icon',
    ELEMENT_LK: 'like_icon',
    ELEMENT_CM: 'comment_icon',
    ELEMENT_LC: 'like_comment_icon',
    ELEMENT_AN: 'app_name',
    ELEMENT_LHL: 'logo_hualala',
    ELEMENT_LGB: 'logo_gubi',
    ELEMENT_NL: 'n_like',
    ELEMENT_NC: 'n_comment',
    ELEMENT_LWD: 'logo_wandou',
    ELEMENT_MWD: 'mpcode_wandou',
    ELEMENT_RWD: 'recode_wandou',
    ELEMENT_RWDI: 'recode_wandou_img',
    ELEMENT_QR: 'qrcode',
    ELEMENT_QRC: 'qrcode_content',
    ELEMENT_LWD1: "logo_wandou_siwei",
    ELEMENT_LWD2: "logo_wandou_koucai",
    ELEMENT_LWD3: "logo_wandou_yizhi",
    ELEMENT_LWD4: "logo_wandou_suzhi",
}


def xname(ename):
    return ELE_NAME.get(ename, None)


ELE_FILTER = set([
    ELEMENT_PT,
    ELEMENT_NN,
    ELEMENT_DS,
    ELEMENT_PS,
    ELEMENT_GP,
    ELEMENT_TM,
    ELEMENT_AN,
    ELEMENT_LHL,
    ELEMENT_LGB,

    ELEMENT_NL,
    ELEMENT_NC,
    ELEMENT_LWD,
    ELEMENT_MWD,
    ELEMENT_RWD,
    ELEMENT_QR,
    ELEMENT_QRC,
    ELEMENT_LWD1,
    ELEMENT_LWD2,
    ELEMENT_LWD3,
    ELEMENT_LWD4,
])

ELE_FILTER_ALL = set([
    ELEMENT_PT,
    ELEMENT_NN,
    ELEMENT_DS,
    ELEMENT_PS,
    ELEMENT_GP,
    ELEMENT_TM,
    ELEMENT_DL,
    ELEMENT_TD,
    ELEMENT_LK,
    ELEMENT_CM,
    ELEMENT_LC,
    ELEMENT_AN,
    ELEMENT_LHL,
    ELEMENT_LGB,
    ELEMENT_NL,
    ELEMENT_NC,
    ELEMENT_LWD,
    ELEMENT_MWD,
    ELEMENT_RWD,
    ELEMENT_QR,
    ELEMENT_QRC,
    ELEMENT_RWDI,
    ELEMENT_LWD1,
    ELEMENT_LWD2,
    ELEMENT_LWD3,
    ELEMENT_LWD4,
])

ELE_FILTER = ELE_FILTER_ALL


def in_filter(element):
    return element.name in ELE_FILTER


def coord_trans(T):
    class C(T):
        @property
        def x1(self):
            return self.cx - 0.5 * self.width

        @property
        def y1(self):
            return self.cy - 0.5 * self.height

        @property
        def x2(self):
            return self.cx + 0.5 * self.width

        @property
        def y2(self):
            return self.cy + 0.5 * self.height

        def inside_hor(self, x, y):
            return y > self.cy - 0.5 * self.height \
                   and y < self.cy + 0.5 * self.height

        def inside_ver(self, x, y):
            return x > self.cx - 0.5 * self.width \
                   and x < self.cx + 0.5 * self.width

        def inside(self, x, y):
            return x > self.cx - 0.5 * self.width \
                   and x < self.cx + 0.5 * self.width \
                   and y > self.cy - 0.5 * self.height \
                   and y < self.cy + 0.5 * self.height

        def world2local(self, rect_world):
            """ 世界坐标 -> 局部坐标 """
            r = rect_world
            # 变换矩阵
            m = np.array([
                [self.width, 0, 0],
                [0, self.height, 0],
                [self.x1, self.y1, 1]
            ])
            m = np.linalg.inv(m)
            # 坐标变换
            x1, y1, _ = np.dot(np.array([r.x1, r.y1, 1]), m)
            x2, y2, _ = np.dot(np.array([r.x2, r.y2, 1]), m)
            return Rect(
                0.5 * (x1 + x2),
                0.5 * (y1 + y2),
                x2 - x1,
                y2 - y1
            )

        def local2world(self, rect_local):
            """ 局部坐标 -> 世界坐标 """
            r = rect_local
            # 变换矩阵
            m = np.array([
                [self.width, 0, 0],
                [0, self.height, 0],
                [self.x1, self.y1, 1]
            ])
            # m = np.linalg.inv(m)
            # 坐标变换
            x1, y1, _ = np.dot(np.array([r.x1, r.y1, 1]), m)
            x2, y2, _ = np.dot(np.array([r.x2, r.y2, 1]), m)
            return Rect(
                0.5 * (x1 + x2),
                0.5 * (y1 + y2),
                x2 - x1,
                y2 - y1
            )
    return C


@coord_trans
class Rect(namedtuple('Rect', 'cx cy width height')): pass


class ElementNone(namedtuple('Element', 'name')):

    @property
    def exists(self):
        return False

    @property
    def xname(self):
        return ELE_NAME[self.name]

    def dumps(self):
        return {'content': None, 'location':{'cx': 0, 'cy': 0, 'w': 0, 'h': 0}}


@coord_trans
class ElementImg(namedtuple('Element', 'name cx cy width height img')):

    def asdict(self):
        return {
            'name': self.name,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height
        }

    @property
    def exists(self):
        return True

    @property
    def xname(self):
        return ELE_NAME[self.name]
    
    def dumps(self):
        return {
            'content': encode_image(self.img),
            'location': {'cx': self.cx, 'cy': self.cy, 'w': self.width, 'h': self.height}
        }


@coord_trans
class ElementTxt(namedtuple('Element', 'name cx cy width height text')):

    def asdict(self):
        return {
            'name': self.name,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height
        }

    @property
    def exists(self):
        return True

    @property
    def xname(self):
        return ELE_NAME[self.name]
    
    def dumps(self):
        return {'content': self.text, 'location': {'cx': self.cx, 'cy': self.cy, 'w': self.width, 'h': self.height}}


@coord_trans
class LabelTuple(namedtuple('LabelTuple', 'label confidence cx cy width height')):
    """
    标签元组
    """
    def iou_ver(tuple1, tuple2):
        """ 垂直方向的交并值 """
        a1 = tuple1.cy - tuple1.height * 0.5
        a2 = tuple1.cy + tuple1.height * 0.5
        b1 = tuple2.cy - tuple2.height * 0.5
        b2 = tuple2.cy + tuple2.height * 0.5
        l1 = a2 - a1
        l2 = b2 - b1
        i = (l1 + l2 - abs(b1 - a1) - abs(b2 - a2)) * 0.5
        # i = max(0, i)
        return i / min(l1, l2)

    def iou_hor(tuple1, tuple2):
        """ 水平方向的交并值 """
        a1 = tuple1.cx - tuple1.width * 0.5
        a2 = tuple1.cx + tuple1.width * 0.5
        b1 = tuple2.cx - tuple2.width * 0.5
        b2 = tuple2.cx + tuple2.width * 0.5
        l1 = a2 - a1
        l2 = b2 - b1
        i = (l1 + l2 - abs(b1 - a1) - abs(b2 - a2)) * 0.5
        # i = max(0, i)
        return i / min(l1, l2)

    def __eq__(self, other):
        return self.label == other.label \
               and abs(self.confidence - other.confidence) < 0.0001 \
               and abs(self.cx - other.cx) < 0.0001 \
               and abs(self.cy - other.cy) < 0.0001 \
               and abs(self.width - other.width) < 0.0001 \
               and abs(self.height - other.height) < 0.0001


class LabelGroup(object):
    """
    标签组
    """
    def __init__(self, tuples=None, group_tuples=None):
        if tuples:
            self.group_tuples = LabelGroup.group_tuple_hor(tuples)
        elif group_tuples:
            self.group_tuples = group_tuples
        else:
            self.group_tuples = []
        self.elements = {}
        self.bottom = None

    def group_tuple_hor(tuples):
        """
        按行聚合标签
        return: array(array(tuple))
        """
        tuples = sorted(tuples, key=lambda v: v.cy)
        group = []
        needle = 0
        while needle < len(tuples):
            tuple1 = tuples[needle]
            g = [tuple1]
            for i in range(needle+1, len(tuples)):
                tuple2 = tuples[i]
                if LabelTuple.iou_ver(tuple1, tuple2) > 0.75 or tuple1.label == tuple2.label:
                    g.append(tuple2)
                    needle += 1
                else:
                    break
            group.append(g)
            needle += 1
        # 每一行去掉重复标签(取置信度最高的)
        group_new = []
        for tuples in group:
            d = {}
            for t in tuples:
                if t.label in d:
                    if t.confidence > d[t.label].confidence:
                        d[t.label] = t
                else:
                    d[t.label] = t
            group_new.append(list(d.values()))
        return group_new

    def get_length(self):
        return len(self.group_tuples)

    def is_empty(self):
        return self.get_length() == 0

    def tuple_iter(self):
        for tuple in chain(*self.group_tuples):
            yield tuple

    def get_rect(self):
        """ 矩形区域 """
        x_min = y_min = sys.maxsize
        x_max = y_max = -sys.maxsize
        for t in self.tuple_iter():
            x1, y1, x2, y2 = t.x1, t.y1, t.x2, t.y2
            x_min = min(x_min, x1)
            y_min = min(y_min, y1)
            x_max = max(x_max, x2)
            y_max = max(y_max, y2)
        return Rect(
            cx = 0.5 * (x_min + x_max),
            cy = 0.5 * (y_min + y_max),
            width = x_max - x_min,
            height = y_max - y_min
        )

    def has_tuple(self, tuple):
        """ 标签是否在集合中 """
        if tuple is None:
            return False
        for tuple0 in chain(*self.group_tuples):
            if tuple.label == tuple0.label:
                if LabelTuple.iou_ver(tuple, tuple0) > 0.9:
                    return True
        return False

    def search_label(self, label):
        """ 搜索标签 """
        t0 = None
        c0 = -1
        for t in self.tuple_iter():
            if t.label == label and t.confidence > c0:
                t0 = t
                c0 = t.confidence
        return t0

    def slice_out(self, start, end):
        """ 切掉一部分 [start, end) """
        return LabelGroup(
            group_tuples=self.group_tuples[0:start] + self.group_tuples[end:]
        )

    def match_temp_one(self, include, exclude, is_floor):
        """ 匹配一个标签模版(匹配到就返回) """
        name_group = list(map(
            lambda v1: list(map(
                lambda v2: v2.label, filter(lambda v3: v3.label not in exclude, v1)
            )), self.group_tuples
        ))
        if len(name_group) == 0:
            return LabelGroup(), None, None
        idx_name_group = zip(list(range(len(name_group))), name_group)
        idx_name_group = list(filter(
            lambda v: len(v[1]) > 0, idx_name_group
        ))
        if len(idx_name_group) == 0:
            return LabelGroup(), None, None
        idxes, name_group = map(list, zip(*idx_name_group))
        s0 = list(map(lambda v: set(v), include))
        for i in range(0, len(name_group)-len(include)+1):
            s1 = list(map(lambda v: set(v), name_group[i:i+len(include)]))
            u = list(filter(
                # lambda v: len(v[0] & v[1]) == len(v[0]) == len(v[1]),     # 双方一模一样
                lambda v: len(v[0] & v[1]) == len(v[0]),                    # 包含模版所有内容
                zip(s0, s1)
            ))
            if len(u) == len(include):
                s = i
                e = i + len(include)
                s = 0 if is_floor else idxes[i]
                e = idxes[e-1] + 1
                return LabelGroup(group_tuples=self.group_tuples[s:e]), s, e
        return LabelGroup(), None, None

    def match_temp_mul(self, temps):
        """ 匹配多个标签模版(匹配到就返回) """
        for temp in temps:
            screen = temp['screen']
            include = temp['include']
            exclude = temp['exclude']
            is_floor = temp['is_floor']
            sub_group, s, e = self.match_temp_one(include, set(exclude), is_floor)
            if not sub_group.is_empty():
                return screen, sub_group, s, e
        return -1, LabelGroup(), None, None

    def match_temp_most(self, temps):
        """ 尽可能的匹配 """
        group = self
        matches = []
        screen = -1
        while True:
            scr, sub_group, start, end = group.match_temp_mul(temps)
            if not sub_group.is_empty():
                screen = scr
                matches.append(sub_group)
                group = group.slice_out(start, end)
            else:
                break
        return screen, matches



if __name__ == '__main__':
    # t1 = LabelTuple(
    #     label='xxoo',
    #     confidence=0.75,
    #     cx=50,
    #     cy=50,
    #     width=100,
    #     height=100
    # )
    # t2 = LabelTuple(
    #     label='xxoo',
    #     confidence=0.75,
    #     cx=50,
    #     cy=50,
    #     width=100,
    #     height=100
    # )
    # print(abs(0.75 - 0.67))

    r1 = Rect(0.5, 0.5, 0.5, 0.5)
    # r2 = Rect(0.5, 0.5, 0.5, 0.5)
    r2 = Rect(0.625, 0.625, 0.25, 0.2)
    
    r3 = r1.world2local(r2)
    r4 = r1.local2world(r3)
    print(r4)
