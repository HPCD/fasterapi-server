#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:validation_rule_processing.py
@time:2021/09/13
处理配置文件中的配置规则,依据配置规则中的优先级进行处理
"""
import logging
import cv2

from src.utils import read_yaml, base64_to_cv_image, get_image_name
from src.element_check.match import match
from src.tools.img_template_match import match_template
import sys
import os

RULE_CONFIG_PATH = os.path.join('config', 'config.yaml')
# 返回状态码，1：通过，2：不通过， 3:不确定
PASS_STATUS_CODE = 1
FAIL_STATUS_CODE = 2
NO_SURE_STATUS_CODE = 3



def verify_result(detect_element_content, limit_content, condition, content_type, upload_time):
    """

    :param detect_element_content: 朋友圈检测内容，需要比较的内容，字符串，图片，数值三种类型
    :param limit_content: 配置规则的限制内容
    :param condition: 比较条件 and : 所有 or: 其中之一  not:不包含  >: 大于    < : 小于
    :param content_type: 内容类型, string, img, number
    :return: True or False
    """

    # logging.debug('verify result cmp content: {}'.format(detect_element_content))
    # 字符串类型
    if content_type == 'string':

        if condition == 'and':
            for word in limit_content:
                if not (word in detect_element_content):
                    return False
            return True
        if condition == 'or':
            for word in limit_content:
                if word in detect_element_content:
                    return True
            return False
        if condition == 'not':
            for word in limit_content:
                if word in detect_element_content:
                    return False
            return True
    # 图像base64 类型
    elif content_type == 'img':
        logging.info("verify img !")

        if condition == 'or':
            img_path = get_image_name()
            s, i = base64_to_cv_image(detect_element_content, img_path)

            result = match(img_path, limit_content)
            poster_status = result['poster']
            logo_status = result['logo']

            if poster_status or logo_status:
                os.remove(img_path)
                return True

            if match_template(img_path, limit_content):
                return True

            return False

    elif content_type == 'number':
        set_content = int(limit_content[0])

        detect_element_content = int(detect_element_content)
        logging.info("detect: {},upload time: {},diff : {}".format(detect_element_content, upload_time,
                                                                   upload_time - detect_element_content))

        # 小于号
        if condition == '<':
            if 0 < upload_time - detect_element_content < set_content:
                return True
            else:
                return False
        # 大于号
        if condition == '>':
            if upload_time - detect_element_content > set_content:

                return True
            else:
                return False


def block_verify_result(block, regulation, upload_time):
    """
    对每一个block进行验证，验证方式：遍历配置文件中设置的所有规则，根据该规则找到block中的元素是否存在，不存在直接验证失败，存在则进行内容上的验证

    block:  {'元素名称':{'content':内容}}， 
            如：{'like_icon': {'content': None}, 'comment_icon': {'content': None}, 'like_comment_icon': {'content': None}}
    
    regulation: {'user_img': {'name': 'user_img', 'if_verify': False, 
                'level': 3}, 'user_name': {'name': 'user_name', 'if_verify': False, 'level': 3}} 
    
    return: True or False 
    """

    if block is None:
        logging.info("block is None")
        # 一个元素都检测不到确定为错误
        return FAIL_STATUS_CODE, 'block is None'

    # 没有配置对应的规则,没有配置规则默认是通过
    if regulation is None:
        logging.info("Regulation is None")
        # 没有规则代表没有限制，所有情况都符合
        return PASS_STATUS_CODE, "Regulation is None"

    logging.info("Begin verify !!!")
    # 遍历配置文件
    for elements_name, rule in regulation.items():
        # 元素名称
        elements_name = rule['name']
        # 是否需要对该元素进行验证，true：验证
        is_verify = rule['if_verify']

        # 配置规则中需要进行验证的元素
        if is_verify:
            # 需要验证的元素分两部分进行：
            # step1 ai是否检测到该元素，并且该元素的content不能为空
            # step2: 依据配置规则条件，逐条比较

            # step1: ai检测不到元素，返回false
            if elements_name not in block.keys():
                return NO_SURE_STATUS_CODE, "{} can not extract".format(elements_name)

            # 找到该元素的内容
            ele_content = block[elements_name]

            # step1: ai检测到的元素content字段为None
            if ele_content['content'] is None:
                logging.info("extract element {} is None".format(elements_name))
                logging.debug("{}".format(block))
                # 分组图标为空的情况，忽略
                if elements_name == 'grouping_icon':
                    continue
                return NO_SURE_STATUS_CODE, "{} is None".format(elements_name)
            # step1: ai检测到的元素content字段不为None,但不能是分组图标
            elif elements_name == 'grouping_icon':

                logging.info("Exists grouping icon")
                return FAIL_STATUS_CODE, "Exists grouping icon"

            # 获取非空的元素content内容
            detect_element_content = ele_content['content']

            # step2: 依据配置规则逐条比较
            # 配置文件中是否有限制条件，限制条件有：and,or,all,not
            if 'limit_condition' in rule.keys():
                limit_condition = rule['limit_condition']
            # 没有限制条件的只需要检测到该元素即可
            else:
                continue

            # 限制内容
            if 'limit_content' in rule.keys():
                limit_content = rule['limit_content']
                logging.info("limit_content : %s", limit_content)
                content_type = rule['content_type']

            # 配置规则中没有设置限定内容，继续比较下一个
            if len(limit_content) == 0 or limit_condition == 'all':
                continue

            logging.info("limit_condition：{}".format(limit_condition))
            logging.info("content_type：{}".format(content_type))
            # 元素验证结果
            verify_result_status = verify_result(detect_element_content, limit_content, limit_condition, content_type,
                                                 upload_time)

            if not verify_result_status:
                logging.info("element {} verify fail .".format(elements_name))
                return NO_SURE_STATUS_CODE, "element {} verify fail .".format(elements_name)
    return PASS_STATUS_CODE, 'success'


def verify_status(user_name, detect_elements, upload_time, rule_num='rule1'):
    """
    status_code: pass: 1, fail: 2 , no sure: 3
    返回验证状态及状态信息
    :param detect_elements: 朋友圈截图上检测到的元素
    :return:
    """

    if detect_elements is None:
        logging.info("user {} : elements is None !".format(user_name))
        return NO_SURE_STATUS_CODE,'other reason'

    # 不同场景序号对应不同的配置规则
    screens = {1: 'regulation_screen1', 2: 'regulation_screen2'}

    # 配置文件
    # config_path = r'D:\project\py-project\ai-wechat-moments-verify\config\config.yaml'
    config = read_yaml(RULE_CONFIG_PATH)


    # 检测场景
    screen_num = detect_elements['screen']
    # -1 表示不是朋友圈,或者不完整的朋友圈截图
    if int(screen_num) == -1:
        return NO_SURE_STATUS_CODE, 'Screen not exists'
    # -2 表示不是朋友圈,没有检测到任何一个元素
    if int(screen_num) == -2:
        return FAIL_STATUS_CODE, 'No wechat img'
    # 场景序号不存在
    if screen_num not in screens.keys():
        logging.info("Screen num: {} not exists".format(screen_num))
        return NO_SURE_STATUS_CODE, "Screen num: {} not exists".format(screen_num)

    regulation = config['user_regulation'][user_name][rule_num][screens[screen_num]]

    # 一张图片可以有不同的block
    blocks_elements = detect_elements['blocks']

    block_verify_status_msg = "blocks elements verify fail !"
    # 对每个block进行验证
    for block in blocks_elements:
        # block 验证状态
        block_verify_status_code, block_verify_status_msg = block_verify_result(block, regulation, upload_time)

        # 任一block符合遍验证通过或出现不通过直接返回
        if block_verify_status_code == PASS_STATUS_CODE or block_verify_status_code == FAIL_STATUS_CODE:
            return block_verify_status_code, block_verify_status_msg

    logging.info("blocks elements verify fail !")
    return NO_SURE_STATUS_CODE, block_verify_status_msg


if __name__ == '__main__':

    predict_elements = {'screen': 1, 'blocks': [[{'element': 'user_img', 'exists': True, 'content': ''},
                                                 {'element': 'user_name', 'exists': True, 'content': ["lucy"]},
                                                 {'element': 'comment', 'exists': True, 'content': ['画啦啦年中大促，一切皆有可能！']},
                                                 {'element': 'share_pictures', 'exists': True, 'content': ['img1']},
                                                 {'element': 'publication_time', 'exists': True, 'content': ['2000']},
                                                 {'element': 'grouping_icon', 'exists': False, 'content': []},
                                                 {'element': 'location', 'exists': True, 'content': ['广州']},
                                                 {'element': 'delete_button', 'exists': True, 'content': []},
                                                 {'element': 'two_point_icon', 'exists': True, 'content': []},
                                                 {'element': 'like_icon', 'exists': True, 'content': []},
                                                 {'element': 'comment_icon', 'exists': False, 'content': []}]]}

    from src.element_check.match import init_match_worker

    init_match_worker()
    predict_elements = {'screen': 1,
                        'blocks': [{'user_name': {'content': None}, 'user_img': {'content': None},
                                    'comment': {'content': ['画啦啦年中大促，一切皆有可能！美术课开课']},
                                    'share_pictures': {'content': ['画啦啦年中大促，一切皆有可能！']},
                                    'publication_time': {'content': 12600},
                                    'app_name': {'content': ['咕比启蒙']},
                                    'grouping_icon': {'content': 'ddd'}
                                    }

                                   ]
                        }
    status = verify_status('gubi', predict_elements)
    if status:
        print("pass")
    else:
        print('failed')
    # print(result)
