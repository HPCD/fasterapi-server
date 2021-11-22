# coding: utf-8

import os
import json
from src.element_check.worker_base import SubWorkerBase, WorkerPoolBase


ZXING_PATH = os.path.realpath(os.path.join('models', 'zxing-tool.jar'))


class WorkerQr(SubWorkerBase):

    def start_proc_cmd(self):
        return ['java', '-jar', ZXING_PATH]

    def proc_cmd(self, img_path):
        """ 加工命令 """
        s = json.dumps({'requestId': 'xxoo', 'imgPath': img_path})
        s += '\n'
        return s.encode('utf-8')

    def proc_res(self, res):
        """ 加工响应 """
        return json.loads(res)


class WorkerQrPool(WorkerPoolBase):
    def __init__(self, n_worker=1, n_res=4):
        super().__init__(WorkerQr, n_worker, n_res)


pool = None

def init_qr_worker():
    global pool
    pool = WorkerQrPool()


def stop_qr_worker():
    global pool
    pool.quit()


def parse_qrcode(img_path):
    global pool
    code, res = pool.exec(img_path=img_path)
    return res if code == 0 else None



if __name__ == '__main__':
    import time
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # img_path = '/Users/penghuan/Documents/liuyi/like-comment/gubi/7a0e731e50acbdd62dc098908db36ea0.jpg'
    # img_path = '/Users/penghuan/Documents/liuyi/like-comment/gubi/3acbe06af49bbeb2043bbc4585faae1d.jpg'
    img_path = '/Users/penghuan/Code/liuyi/ai-wechat-moments-verify/img/poster_tmp_99.jpg'

    pool = WorkerPoolBase(WorkerQr, 1, 2)
    code, res = pool.exec(img_path=img_path)
    print(code, res)
    pool.quit()

    # import subprocess
    # proc = subprocess.Popen(
    #     ['java', '-jar', ZXING_PATH],
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    #     bufsize=0
    # )

    # time.sleep(10)

