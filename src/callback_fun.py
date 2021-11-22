#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:callback_fun.py
@time:2021/10/26
"""
import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
from urllib.parse import urljoin
import os
from src.utils import send_dd_warning

# SEND_DingDing_INFO = True
def send_predict_result_to_callback(result,user_name,service='algo-wechat-moments-verify',api="/i/autoComment/v1.0/callback",send_dingding_info=False,dingdingurl=''):
    """
    发送结果给回调接口
    Args:
        result:
        service:

    Returns:

    """
    # service = 'arts-class-intelligent-scoring'
    # 获取回调服务的host IP
    host = get_service_host(service,send_dingding_info,dingdingurl)

    send_url = host + api
    logging.info("Step3: callback api %s",send_url)
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=5))  # 设置重试次数
    s.mount('https://', HTTPAdapter(max_retries=5))
    try:
        logging.info("Step3 : save result %s",result)
        res = requests.post(send_url, json=result, timeout=10)
        if res.status_code == 200:
            logging.info("success")
            result = json.loads(res.text)
            code = int(result['code'])
            logging.info("Step3: code is %s",code)
            msg = str(result['msg'])
            logging.info("Step3: callback msg %s",msg)

            if code != 0:
                info = "{}回调接口错误, 错误信息：{}".format(user_name,msg)
                if send_dingding_info:
                    send_dd_warning(info,dingdingurl)
        else:
            info = "{} 回调接口错误,code {}".format(user_name,res.status_code)
            if send_dingding_info:
                send_dd_warning(info,dingdingurl)
    except requests.exceptions.ConnectionError as e:
        logging.info('{}回调接口返回失败 ！！!'.format(user_name))
        # 发送钉钉警告
        info = "{} 回调接口错误，服务不存在 ！".format(user_name)
        if send_dingding_info:
            send_dd_warning(info,dingdingurl)



class ServiceNotFoundError(Exception):
    """ 在eureka找不到服务时抛出这个异常 """
    pass


class NoInstanceError(Exception):
    """ 服务没有实例，抛出这个异常 """
    pass


def get_service_host(service,send_dingding_info=False,dingding_url=None):
    """
    从eureka获取指定服务的ip和端口，如果有多个节点，只返回第一个
    :param service: 注册到eureka服务名，如 draw-course-backend
    :return: 第一个节点的服务ip和端口，如 http://172.16.51.6:8889/
    """

    # eureka地址,环境变量获取
    eureka_url = os.environ.get("EUREKA_SERVER", default="http://config.dev.61info.com:8761/")
    logging.info("Get service host: eureke_url is {}".format(eureka_url))
    active = eureka_url

    # 防止建立过多连接，使用session来请求接口
    session = requests.Session()
    session.keep_alive = False

    # 最多重试3次，间隔0.5s
    adapter = HTTPAdapter(max_retries=Retry(connect=3, backoff_factor=0.5))
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    headers = {'Accept': 'application/json'}
    res = session.get(urljoin(active, '/eureka/apps/%s' % service), headers=headers)
    if res.status_code != 200:
        logging.info('The  service maybe not be registered in eureka!')
        info = "注册中心未发现回调接口服务"
        if send_dingding_info:
            send_dd_warning(info,dingding_url)
        raise ServiceNotFoundError('The "%s" service maybe not be registered in eureka!' % service)
    content = str(res.content, encoding='utf-8')
    content = json.loads(content)
    instance = content['application']['instance']
    if not instance:
        info = "回调接口注册中心服务不可用！"
        # 发送钉钉消息
        if send_dingding_info:
            send_dd_warning(info,dingding_url)
        raise NoInstanceError('No "%s" service instance available' % service)
    instance = instance[0]
    logging.info("Return homePageUrl ： {}".format(instance['homePageUrl']))

    return instance['homePageUrl']