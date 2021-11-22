#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:handle_eureka.py
@time:2021/09/09
"""
import py_eureka_client.eureka_client as eureka_client
import py_eureka_client.netint_utils as netint_utils
import logging

def on_err(err_type: str, err: Exception):
    if err_type in (eureka_client.ERROR_REGISTER, eureka_client.ERROR_DISCOVER):
        eureka_client.stop()
    else:

        logging.info("err_tyep : %s ", err_type)
        logging.info("err : %s ", err)


def set_eureka(eureka_server="http://config.dev.61info.com:8761/eureka", server_port=32666,
               app_name='arts-class-intelligent-scoring'):
    """
    注册eureka 给到java调用
    eureka : eureka有开发环境有正式环境
    :return:
    """
    # server_host = "localhost"
    logging.info("Eureka register")
    logging.info("Register server %s", eureka_server)
    logging.info("Register server port %s", server_port)
    ip, server_host = netint_utils.get_ip_and_host("10.0.0.0/8")

    client = eureka_client.init(eureka_server=eureka_server,
                                app_name=app_name,
                                # 当前组件的主机名，可选参数，如果不填写会自动计算一个，如果服务和 eureka 服务器部署在同一台机器，请必须填写，否则会计算出 127.0.0.1
                                instance_host=ip,
                                instance_port=server_port,
                                instance_ip=ip,
                                # 调用其他服务时的高可用策略，可选，默认为随机
                                ha_strategy=eureka_client.HA_STRATEGY_RANDOM,
                                on_error=on_err)
    client.status_update("UP")
    logging.info("Success Register Eureka !")

