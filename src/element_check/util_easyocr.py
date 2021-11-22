# coding: utf-8

import os
import cv2
import easyocr


MODEL_DIR = os.path.realpath('models')
IMAGE_DIR = os.path.realpath('img')


easy_reader_en = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory=MODEL_DIR,
    download_enabled=False
)


easy_reader_ch = easyocr.Reader(
    ['ch_sim', 'en'],
    gpu=False,
    model_storage_directory=MODEL_DIR,
    download_enabled=False
)


def extract_text(id, img, ch=False):
    tmp_path = os.path.join(IMAGE_DIR, 'easy_ocr_%d.jpg' % id)
    cv2.imwrite(tmp_path, img)
    reader = easy_reader_ch if ch else easy_reader_en
    result = reader.readtext(tmp_path)

    h0, w0 = img.shape[:2]
    ret = []
    for pos, txt, confidence in result:
        x1, y1 = pos[0]
        x2, y2 = pos[2]
        ret.append({
            'text': txt,
            'cx': 0.5 * (x1 + x2) / w0,
            'cy': 0.5 * (y1 + y2) / h0,
            'w': (x2 - x1) / w0,
            'h': (y2 - y1) / h0,
            'confidence': confidence
        })
    return ret



if __name__ == '__main__':
    img_path = '/Users/penghuan/Desktop/222.jpg'
    img = cv2.imread(img_path)
    ret = extract_text(0, img, ch=True)
    print(ret)

