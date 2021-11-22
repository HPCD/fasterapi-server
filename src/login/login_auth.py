#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:login_auth.py
@time:2021/09/10
登录验证
获取token
"""
import os
from flask import make_response,jsonify
from flask_httpauth import HTTPTokenAuth,HTTPBasicAuth
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from itsdangerous import BadSignature, SignatureExpired
import re
import logging
from src.utils import read_yaml

# 密钥
SECRET_KEY = 'wechat-verify@123'
# token 有效时间,单位秒
EXPIRATION = 3600

USER_CONFIG_PATH = os.path.join('config', 'user_info.yaml')

token_auth = HTTPTokenAuth()
basic_auth = HTTPBasicAuth()


@basic_auth.verify_password
def verify_password(user_name, password):
    """
    验证 用户名 密码
    :param user_name:
    :param password:
    :return:
    """

    logging.info('verify login !!!')
    logging.info("user_name: {} password: {}".format(user_name, password))

    user_info = read_yaml(USER_CONFIG_PATH)
    user_list = user_info['UserList']
    # 判断用户是否存在
    for user, _ in user_list.items():
        # 用户名存在，生成token
        if user_name == user_list[user]['user_name'] and password == user_list[user]['password']:
            return True
    return False

@token_auth.verify_token
def verify_token(token):

    logging.info("verify auth token %s", token)
    s = Serializer(SECRET_KEY)
    try:
        # 解密
        data = s.loads(token)
        logging.info('token verify success !!! %s', data)
        return data
    except SignatureExpired:
        logging.info("Token 过期 ！")
        return None
    except BadSignature:
        logging.info("Token 不正确！")
        return None

def generate_token(user_name):


    logging.info("user name exists !")
    logging.info('generate_auth_token')
    # SECRET_KEY
    s = Serializer(SECRET_KEY, expires_in=EXPIRATION)
    return s.dumps({'user_name': user_name})

@token_auth.error_handler
def token_auth_error():
    """
    自定义token验证失败返回信息
    :return:
    """
    return make_response(jsonify({'code':'E004','message':'token time out or wrong !','success': False}),401)

@basic_auth.error_handler
def login_auth_error():
    """
    自定义登录失败返回信息
    :return:
    """
    return make_response(jsonify({'code':'E001','success':False,'message':'错误的用户名或密码'}),401)










if __name__ == '__main__':

    pass
