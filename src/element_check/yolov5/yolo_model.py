# coding: utf-8

import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(__file__)))

import cv2
import torch
import numpy as np
from urllib.request import Request, urlopen

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, time_sync
from utils.plots import colors, plot_one_box


def download(url, to=None):
    try:
        request = Request(url)
        response = urlopen(request)
        raw = response.read()
        if to is not None:
            with open(to, 'wb') as f:
                f.write(raw)
        print('Downloaded from %s, save to %s' % (url, to,))
        return raw
    except Exception as err:
        print('Download Fail, %s, %s' % (err, url,), file=sys.stderr)
    return None


class YoloModel(object):
    def __init__(self, weight, device=''):
        self.device = select_device(device)
        self.half = False
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        self.stride, self.labels = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.model = attempt_load(weight, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.labels = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.labels2id = dict(list(map(
            lambda v: (v[1], v[0]),
            enumerate(self.labels)
        )))
        if self.half:
            self.model.half()  # to FP16
        self.clear_img()

    def _read_img(self, source):
        """
        source: image source, url or local
        """
        if source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://')
        ):
            raw = download(source)  # download from remote
            # raw = None  # deprecate downloading from remote
            assert raw is not None
            arr = np.asarray(bytearray(raw), dtype=np.uint8)
            img0 = cv2.imdecode(arr, -1)
        else:
            img0 = cv2.imread(source)
        return img0

    def _format_img(self, img0, imgsz=640):
        """
        img0: origin image
        imgsz: inference size (pixels)
        """
        # Padded resize
        img = letterbox(img0, imgsz, stride=self.stride)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16)
        # Format
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def load_img(self, source, imgsz=640):
        """
        source  : image source, url or local
        imgsz   : inference size (pixels)
        """
        t1 = time_sync()
        self.clear_img()
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.img_ori = self._read_img(source)
        assert self.img_ori is not None
        self.img_fmt = self._format_img(self.img_ori, imgsz)
        self.img_pre = self.img_ori.copy()
        t2 = time_sync()
        print(f'Load image done. ({t2 - t1:.2f}s)')

    def save_img(self, path_ori=None, path_pre=None):
        if path_ori is not None and self.img_ori is not None:
            cv2.imwrite(path_ori, self.img_ori)
        if path_pre is not None and self.img_pre is not None:
            cv2.imwrite(path_pre, self.img_pre)

    def get_img_shape(self):
        if self.img_ori is not None:
            shape = self.img_ori.shape
            return {'width': shape[1], 'height': shape[0]}
        return None

    def clear_img(self):
        self.img_ori = None     # origin image
        self.img_fmt = None     # formate image
        self.img_pre = None     # predict image

    # 裁剪
    def crop_img(self, cx, cy, width, height, **kwargs):
        shape = self.get_img_shape()
        cx = cx * shape['width']
        cy = cy * shape['height']
        width = width * shape['width']
        height = height * shape['height']
        x1 = int(cx - width * 0.5)
        y1 = int(cy - height * 0.5)
        x2 = int(cx + width * 0.5)
        y2 = int(cy + height * 0.5)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(shape['width']-1, x2)
        y2 = min(shape['height']-1, y2)
        img_crop = self.img_ori[y1:y2, x1:x2]
        if 'resize_shape' in kwargs:
            img_crop = cv2.resize(
                img_crop, 
                kwargs['resize_shape'], 
                # interpolation=cv2.INTER_CUBIC
            )
        return img_crop

    @torch.no_grad()
    def pred(self,
             conf_thres=0.25,
             iou_thres=0.45,
             max_det=1000,
             labels=None,
             agnostic_nms=False,
        ):
        """
        args
            conf_thres      : confidence threshold
            iou_thres       : NMS IOU threshold
            max_det         : maximum detections per image
            labels          : filter by labels, [string]
            agnostic_nms    : class-agnostic NMS
        return
            [
                {
                    'label': string, 
                    'conf': float, 
                    'cx': float, 
                    'cy': float, 
                    'width': float, 
                    'height': float
                }
            ...]
        """
        # Inference
        t1 = time_sync()
        pred = self.model(self.img_fmt)[0]
        # NMS
        classes = None
        if labels is not None:
            classes = []
            for lb in labels:
                if lb in self.labels2id:
                    classes.append(self.labels2id[lb])
                else:
                    print(f'Label [{lb}] not found.')
        pred = non_max_suppression(
            pred, 
            conf_thres, 
            iou_thres, 
            classes, 
            agnostic_nms, 
            max_det=max_det
        )
        # Process predictions
        ret = []
        gn = torch.tensor(self.img_ori.shape)[[1, 0, 1, 0]]
        for _, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im_ori size
                det[:, :4] = scale_coords(
                    self.img_fmt.shape[2:], 
                    det[:, :4], 
                    self.img_ori.shape
                ).round()
                for *xyxy, conf, c in det:
                    c = int(c)
                    conf = conf.item()
                    label = self.labels[c]
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    plot_one_box(
                        xyxy, 
                        self.img_pre, 
                        label=f'{label} {conf:.2f}', 
                        color=colors(c, True), 
                        line_thickness=3
                    )
                    ret.append({
                        'label': label,
                        'confidence': conf,
                        'cx': xywh[0],
                        'cy': xywh[1],
                        'width': xywh[2],
                        'height': xywh[3]
                    })
        t2 = time_sync()
        print(f'Predict done. ({t2 - t1:.2f}s)')
        return ret



if __name__ == '__main__':
    model = YoloModel('models/moment_check_20211009.pt')
    model.load_img('/Users/penghuan/Desktop/333.jpeg')
    pred = model.pred()
    print(pred)
