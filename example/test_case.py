#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:test_case.py
@time:2021/11/22
"""
import requests
import json
img_url = 'https://hll-cdn.oss-accelerate.aliyuncs.com/prod/hll-activity/a37fabb8e8e26ed05973ee087f034fd6'
url = 'http://127.0.0.1:32666/faster-server/v1/test/'
data = {'image':img_url,'image_type':'URL'}
content = open(r'E:\dataset\friends\wandou\images\logo_20211109\test\0e0bd1017d0941b0b1d1c66b72d5a523.png','rb')
rsp = requests.post(url,json=data)
print(rsp.json())
# rsp = requests.post(url)
# print(rsp.json())