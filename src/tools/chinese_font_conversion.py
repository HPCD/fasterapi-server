#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:abner
@file:chinese_font_conversion.py
@time:2021/09/16
"""
from src.tools.langconv import *

# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line


# 转换简体到繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line


if __name__ == "__main__":
    line_cht = '把中文字符串進行繁體和簡體中文的轉換'
    line = cht_to_chs(line_cht)
    print(line)
