#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:asyn_methods.py
@time:2021/10/26
异步方法
"""
import logging
import os
from src.utils import get_upload_time, get_image_name, get_img, read_yaml
from src.validation_rule_processing import verify_status
from src.element_check.extract import extract_element,check_gubi
from src.callback_fun import send_predict_result_to_callback
from  time import time
USER_CONFIG_PATH = os.path.join('config', 'user_info.yaml')


def async_veri(save_img_name, user_name, params_data):
    """
    异步审核
    Args:
        save_img_name:
        user_name:
        params_data:

    Returns:

    """

    # 请求参数中获取上传时间
    upload_time = get_upload_time(params_data)
    # 检测基本元素,ts默认当前时间
    logging.info("Step 3: user name : {} ,start extract element! ".format(user_name))
    start_time = time()
    detect_elements = extract_element(save_img_name, upload_time)

    end_time = time()
    logging.info(" Extract element user time : {}".format(end_time-start_time))
    logging.info("Step 3: user name : {} ,finished extract element! ".format(user_name))
    # 检测完毕，删除图片
    if os.path.exists(save_img_name):
        os.remove(save_img_name)

    # 验证检测到的元素状态，审核通过返回true, 否则返回false
    vec_status, verify_msg = verify_status(user_name, detect_elements, upload_time)

    logging.info("Step 3 : Finished verify !")

    result = {
        "imageId": params_data['image_id'],
        "imageUrl": params_data['image'],
        "machineCheckResult": verify_msg,
        "machineCheckStatus": vec_status
    }

    logging.info("Result {}".format(result))

    service, api, send_dingding_info, ding_url = get_eureka_by_username(user_name)

    if service is None :
        logging.info("Can not Find user ：{}".format(user_name))

    logging.info(" Save result to callback !")
    # 结果保存到回调接口
    send_predict_result_to_callback(result, user_name,service, api, send_dingding_info, ding_url)
    logging.info("Step5: User {} Success send result to callback function !".format(user_name))


def gubi_async_veri(user_name, params_data):
    """
    异步审核
    Args:
        save_img_name:
        user_name:
        params_data:

    Returns:

    """

    # 请求参数中获取上传时间
    logging.info("咕比检测！")
    logging.info(params_data)
    start_time = time()
    # 检测基本元素,ts默认当前时间
    try:
        result = check_gubi(**params_data)
        end_time = time()
        logging.info("Gubi user time : {}".format(end_time-start_time))
    except Exception as e:
        logging.info("Gubi check exception: {}!".format(e))


    logging.info("Step 3 : Finished verify !")


    logging.info("Gubi Result {}".format(result))

    service, api,send_dingding_info,ding_url = get_eureka_by_username(user_name)

    if service is None :
        logging.info("Can not Find user ：{}, service is none".format(user_name))

    if not send_dingding_info:
        logging.info('Not send dingding info !')

    logging.info(" Save result to callback !")
    # 结果保存到回调接口
    send_predict_result_to_callback(result,user_name, service, api,send_dingding_info,ding_url)
    # logging.info("回调接口时间： {}".format(time()))
    logging.info("Success send result to callback function !")


def get_eureka_by_username(user_name):
    """
    通过用户名从配置文件中查找到eureka 服务
    Args:
        user_name:

    Returns:

    """

    user_info_config = read_yaml(USER_CONFIG_PATH)
    user_list = user_info_config['UserList']
    service, api = None, None
    send_dingding_info = False
    dingding_url = None
    for user, _ in user_list.items():
        # 用户名存在，生成token
        if user_name == user_list[user]['user_name']:
            callback_fun_config = user_list[user]['callback_fun']
            # 告警机器人
            robot_config = user_list[user]['robot']

            service = callback_fun_config['eureka_service']
            api = callback_fun_config['api']

            send_dingding_info = robot_config['send_dingding_info']
            dingding_url = robot_config['url']
            logging.info("service: {}".format(service))
            break

    logging.info("Step3 : service is :{}".format(service))
    logging.info("Step3 : api : {}".format(service))
    return service, api,send_dingding_info,dingding_url

if __name__ == "__main__":
    service,api,send_dingding_info,ding_url = get_eureka_by_username('hualala1d')
    print(service,api,send_dingding_info,ding_url)