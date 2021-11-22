# coding: utf-8

from src.element_check.worker_base import WorkerBase, WorkerPoolBase


class WorkerMat(WorkerBase):
    def __init__(self, id, *args, **kwargs):
        super().__init__(id, *args, **kwargs)

    def before_run(self):
        from src.element_check.check import ImageMatcher
        self.image_matcher = ImageMatcher(self.id)

    def do(self, **kwargs):
        return self.image_matcher.match(**kwargs)


class WorkerPoolMat(WorkerPoolBase):
    def __init__(self, n_worker=1, n_res=16):
        super().__init__(WorkerMat, n_worker, n_res)


pool = None


def init_match_worker():
    global pool
    pool = WorkerPoolMat()


def stop_match_worker():
    global pool
    pool.quit()


def _match(img_path_match, img_path_temps, img_path_ex_temps, half):
    """
    匹配模版
    img_path_match: 匹配图片路径
    img_path_temps: 模版图片路径列表
    img_path_ex_temps: 需要排除的模版图片路径列表
    half: 是否需要将模版图片截半
    return: True or False
    """
    global pool
    all_path_temps = []
    all_path_temps.extend(img_path_temps)
    all_path_temps.extend(img_path_ex_temps)
    matched_path = None
    matched_score = 0
    for pt in all_path_temps:
        code, res = pool.exec(
            img_path_match=img_path_match, 
            img_path_temp=pt,
            half=half
        )
        if code == 0 and res is not None:
            if res > matched_score:
                matched_path = pt
                matched_score = res
    if matched_path is not None and matched_path in img_path_temps:
        return True
    return False


def _match_poster(img_path_match, img_path_temps, img_path_ex_temps):
    return _match(img_path_match, img_path_temps, img_path_ex_temps, True)


def _match_logo(img_path_match, img_path_temps, img_path_ex_temps):
    return _match(img_path_match, img_path_temps, img_path_ex_temps, False)


def match(img_path, temp_conf):
    """
    图片模版匹配
    img_path: 匹配图片路径
    temp_conf: 模版图片配置
    return: {"poster": bool, "logo": bool}
    """
    temp_poster_in = temp_conf['poster']['include']
    temp_poster_ex = temp_conf['poster']['exclude']
    temp_logo_in = temp_conf['logo']['include']
    temp_logo_ex = temp_conf['logo']['exclude']
    poster = False
    logo = False
    if len(temp_poster_in) + len(temp_poster_ex) > 0:
        poster = _match_poster(img_path, temp_poster_in, temp_poster_ex)
    if len(temp_logo_in) + len(temp_logo_ex) > 0:
        logo = _match_logo(img_path, temp_logo_in, temp_logo_ex)
    return {'poster': poster, 'logo': logo}


if __name__ == '__main__':
    import time
    import cv2
    import cv_app as cva
    from extract import init_extract_worker, extract_element, stop_extract_worker

    img_path = '/Users/penghuan/Code/liuyi/ai-wechat-moments-verify/val/gubi/60f2102f4bc7d80431f38f87.jpeg'

    init_extract_worker()
    init_match_worker()

    res_ele = extract_element(img_path, int(time.time()))

    for block in res_ele['blocks']:
        poster = block['share_pictures']
        if poster['content'] is not None:
            img = cva.decode_image(poster['content'])
            cv2.imwrite('./img/test_match.jpg', img)
        
            res_match = match(
                './img/test_match.jpg',
                {
                    'poster': {
                        'include': [
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_01.png',
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_02.png',
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_03.png',
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_04.png',
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_05.jpeg',
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_06.png',
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_07.jpeg',
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_08.jpeg'
                        ],
                        'exclude': [
                            # '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_01.png',
                            # '/Users/penghuan/Documents/liuyi/gubi-wechat/poster/gubi_tmp_poster_07.jpeg'
                        ]
                    },
                    'logo': {
                        'include': [
                            '/Users/penghuan/Documents/liuyi/gubi-wechat/logo/gubi_tmp_logo_01.jpg'
                        ],
                        'exclude': []
                    }
                }
            )
            print(res_match)

    stop_extract_worker()
    stop_match_worker()

