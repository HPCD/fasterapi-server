# coding: utf-8

import sys
import time

import src.element_check.label as lb
from src.element_check.worker_base import WorkerBase, WorkerPoolBase


class WorkerExt(WorkerBase):
    def __init__(self, id, *args, **kwargs):
        super().__init__(id, *args, **kwargs)

    def before_run(self):
        from src.element_check.check import LabelChecker
        self.label_checker = LabelChecker(self.id)

    def do(self, **kwargs):
        return self.label_checker.check(**kwargs)

    def before_quit(self):
        self.label_checker.quit()


class WorkerPoolExt(WorkerPoolBase):
    def __init__(self, n_worker=1, n_res=16):
        super().__init__(WorkerExt, n_worker, n_res)


pool = None

def init_extract_worker():
    global pool
    pool = WorkerPoolExt()


def stop_extract_worker():
    global pool
    pool.quit()


def extract_element(img_source, ts):
    """
    抽出图片各个元素
    img_source: string 图片源(url or local)
    ts: int 图片上传时间戳
    return: {
        "screen": int,  # 场景类型. -1: 不是朋友圈场景; 1: 场景一; 2: 场景二.
        "blocks": [     # 分块
            {
                # 元素名称: {"content": 时间戳 | 文本内容 | 图片内容}
                string: {"content": int | string | bytes},
                ...
            },
            ...
        ]
    } or None
    """
    global pool
    code, res = pool.exec(img_source=img_source, ts=ts)
    if code != 0:
        return None
    return res


# 二维码类型(咕比AI)
gubi_qr_type = {
    'webapp-ai.61info.cn': "周周分享海报",
    'ai-h5.61info.cn': "学习报告"
}

# 不通过的情况-基本元素(咕比AI)
gubi_base_unpass_conf = [
    {
        # 场景
        'screen': [1],
        # 存在元素
        'ele_yes': [lb.ELEMENT_GP],
        # 不存在元素
        'ele_no': [lb.ELEMENT_DS],
        # 时间限制
        'time_min': None,
        # 不通过原因
        'msg': "没有邀请语, 并且设置了分组"
    },
    {
        'screen': [1],
        'ele_yes': [],
        'ele_no': [lb.ELEMENT_DS],
        'time_min': None,
        'msg': "没有邀请语"
    },
    {
        'screen': [1],
        'ele_yes': [lb.ELEMENT_GP],
        'ele_no': [],
        'time_min': None,
        'msg': "朋友圈设置了分组"
    },
    {
        'screen': [1, 2],
        'ele_yes': [lb.ELEMENT_TM],
        'ele_no': [],
        'time_min': 7200,
        'msg': "截图的朋友圈还未保留 2 小时"
    },
]


# 不通过的情况-海报(咕比AI)
gubi_poster_unpass_conf = {
    1: {
        "name": "学习报告",
        "url_key": "ai-h5.61info.cn",
        "location": {
            "ele_1": lb.ELEMENT_LGB,
            "ele_2": lb.ELEMENT_QR,
            "iou_hor": [-1000.0, -0.5]
        }
    },
    2: {
        "name": "周周分享海报",
        "url_key": "webapp-ai.61info.cn",
        "location": {
            "ele_1": lb.ELEMENT_LGB,
            "ele_2": lb.ELEMENT_QR,
            "iou_hor": [0.7, 1.0]
        }
    },
}


def check_gubi(**kwargs):
    """ 咕比打卡审核 """
    ret = {
        'clock_id': None,
        'check_status': 1,
        'check_info': '',
        'check_time': None
    }
    clock_id = kwargs.get('clock_id', None)
    poster_type = kwargs.get('poster_type', None)
    img_source = kwargs.get('image', None)
    check_desc = kwargs.get('check_desc', True)
    timestamp = kwargs.get('upload_time', int(time.time()))
    ret['clock_id'] = clock_id

    res = extract_element(img_source=img_source, ts=timestamp)
    ret['check_time'] = int(time.time())
    if res is not None and res['screen'] > 0:
        # 搜索咕比区块
        block = None
        for b in res['blocks']:
            if b[lb.xname(lb.ELEMENT_LGB)]['content'] is not None:
                block = b
                break
            elif b[lb.xname(lb.ELEMENT_QRC)]['content'] is not None:
                if b[lb.xname(lb.ELEMENT_QRC)]['content'] in gubi_qr_type:
                    block = b
                    break
        # 没有找到咕比区块
        if block is None:
            ret['check_status'] = 2
            ret['check_info'] = "不是咕比海报"
        else:
            if block[lb.xname(lb.ELEMENT_TM)]['content'] is None:
                block[lb.xname(lb.ELEMENT_TM)]['content'] = 0
            # 基本元素检查
            for conf in gubi_base_unpass_conf:
                ele_yes = conf['ele_yes']
                ele_no = conf['ele_no'] if check_desc else list(filter(lambda e: e != lb.ELEMENT_DS, conf['ele_no']))
                time_min = conf['time_min']
                if not ele_yes and not ele_no and not time_min:
                    continue
                if not time_min:
                    time_min = sys.maxsize
                if res['screen'] in conf['screen'] \
                   and len(list(filter(lambda e: block[lb.xname(e)]['content'] is not None, ele_yes))) == len(ele_yes) \
                   and len(list(filter(lambda e: block[lb.xname(e)]['content'] is None, ele_no))) == len(ele_no) \
                   and (ret['check_time'] - block[lb.xname(lb.ELEMENT_TM)]['content'] < time_min) \
                :
                    ret['check_status'] = 2
                    ret['check_info'] = conf['msg']
                    break
            # 海报类型检查
            if ret['check_status'] != 1:
                conf = gubi_poster_unpass_conf[poster_type]
                ele_1 = block[lb.xname(conf['location']['ele_1'])]
                ele_2 = block[lb.xname(conf['location']['ele_2'])]
                ele_qrc = block[lb.xname(lb.ELEMENT_QRC)]
                poster_passed = False
                if ele_qrc['content'] is not None and conf['url_key'] in ele_qrc['content']:
                    poster_passed = True
                elif ele_1['content'] is not None and ele_2['content'] is not None:
                    loc1 = ele_1['location']
                    loc2 = ele_2['location']
                    r1 = lb.Rect(loc1.cx, loc1.cy, loc1.w, loc1.h)
                    r2 = lb.Rect(loc2.cx, loc2.cy, loc2.w, loc2.h)
                    iou = lb.LabelTuple.iou_hor(r1, r2)
                    if iou >= conf['location']['iou_hor'][0] and iou <= conf['location']['iou_hor'][1]:
                        poster_passed = True
                if not poster_passed:
                    ret['check_status'] = 2
                    ret['check_info'] = "不是%s" % conf['name']
    else:
        ret['check_status'] = 2
        ret['check_info'] = "不是朋友圈场景"

    return ret


if __name__ == '__main__':
    import time
    import cv2
    import src.element_check.label as lb
    import src.element_check.cv_app as cva
    from src.element_check.qrcode import init_qr_worker, stop_qr_worker, parse_qrcode

    img_path = '/Users/penghuan/Documents/liuyi/gubi-wechat/gubi-wechat-test/610156f44bc7d867920ea768.jpeg'
    poster_path = 'img/tmp.jpg'

    init_extract_worker()
    init_qr_worker()

    # ret = extract_element(img_path, int(time.time()))
    ret = check_gubi(image=img_path, check_desc=True, poster_type=2)
    print(ret)

    stop_qr_worker()
    stop_extract_worker()

