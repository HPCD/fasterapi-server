# coding: utf-8

import re
import requests


def get_all_image(url):
    resp = requests.get(url)
    if resp.status_code == 200:
        pattern = re.compile(r'href="(.*)"')
        result = pattern.findall(resp.content.decode('utf-8'))
        return result
    else:
        return []


def get_one_image(url, regex):
    pattern = re.compile(regex)
    for img_name in get_all_image(url):
        if pattern.search(img_name):
            return img_name
    return None


def download_image(url, img_name, to):
    url = '%s/%s' % (url, img_name)
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(to, 'wb') as f:
            f.write(resp.content)
            return True
    return False



if __name__ == '__main__':
    import os

    url = 'http://10.200.11.244:8000/wandou-wechat/images/'
    img_name = get_one_image(url, r'6089001')
    if img_name:
        name, fmt = img_name.split('.')
        print(name, fmt)
        download_image(url, img_name, os.path.join('img', 'tmp.%s' % fmt))
