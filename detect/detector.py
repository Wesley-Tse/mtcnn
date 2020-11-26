#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/26 11:50
import os
import time
import torch
import utils
import numpy as np
from PIL import Image
from Module.module import PNet, RNet, ONet
from torchvision import transforms


class Detector:
    def __init__(self, p_params, r_params, o_params, device):
        self.device = device
        self.p_net = PNet().to(device)
        self.r_net = RNet().to(device)
        self.o_net = ONet().to(device)

        if os.path.exists(p_params):
            self.p_net.load_state_dict(torch.load(p_params))
        if os.path.exists(r_params):
            self.r_net.load_state_dict(torch.load(r_params))
        if os.path.exists(o_params):
            self.o_net.load_state_dict(torch.load(o_params))

        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()

        self.__trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])

    def detect(self, image):
        start_time = time.time()
        p_box = self.__p_net(image)
        if p_box.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        p_time = end_time - start_time
        print('p_time', p_time)
        print('p_net boxes:', p_box.shape)

        start_time = time.time()
        r_box = self.__r_net(image, p_box)
        if r_box.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        r_time = end_time - start_time
        print('r_time', r_time)
        print('r_net boxes:', r_box.shape)

        start_time = time.time()
        o_box = self.__o_net(image, r_box)
        if o_box.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        o_time = end_time - start_time
        total_time = p_time + r_time + o_time
        print('o_time', o_time)
        print('o_net boxes:', o_box.shape)
        print('total_time', total_time)

        return o_box

    def __p_net(self, image):
        boxes = []

        w, h = image.size
        min_side = min(w, h)
        scale = 1.
        while min_side > 12:
            img = self.__trans(image).to(self.device)
            img.unsqueeze_(0)  # N = 1
            with torch.no_grad():
                pre_conf, pre_off = self.p_net(img)
            confidence = pre_conf[0][0].cpu()
            offset = pre_off[0].cpu()
            index = torch.nonzero(torch.gt(confidence, 0.7), as_tuple=False)
            box = self.__box(index, offset, confidence[index[:, 0], index[:, 1]], scale)
            scale *= 0.7
            w = int(w * scale)
            h = int(h * scale)

            image = image.resize((w, h), Image.ANTIALIAS)
            min_side = min(w, h)
            box_ = utils.nms2(box, 0.7, softnms=True)
            boxes.extend(box_)
        return np.stack(utils.nms2(np.array(boxes), 0.6, softnms=True))

    def __r_net(self, image, p_boxes):
        images = []
        p_box = utils.expand_box(p_boxes)  # 将p_net输出的框补成方形
        for box in p_box:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])

            img = image.crop((_x1, _y1, _x2, _y2)).resize((24, 24), Image.ANTIALIAS)
            images.append(self.__trans(img))
        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            pre_conf, pre_off = self.r_net(images)
        confidence = pre_conf.cpu()
        offset = pre_off.cpu()
        boxes = []
        indexs, _ = np.where(confidence > 0.85)
        for index in indexs:
            box = p_box[index]
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])

            w = _x2 - _x1
            h = _y2 - _y1

            x1 = _x1 + w * offset[index][0]
            y1 = _y1 + h * offset[index][1]
            x2 = _x2 + w * offset[index][2]
            y2 = _y2 + h * offset[index][3]

            boxes.append([x1, y1, x2, y2, confidence[index][0]])

        boxes = np.array(torch.tensor(boxes))
        return utils.nms2(boxes, 0.5, softnms=True)

    def __o_net(self, image, r_boxes):
        images = []
        r_box = utils.expand_box(r_boxes)

        for box in r_box:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])

            img = image.crop((_x1, _y1, _x2, _y2)).resize((48, 48), Image.ANTIALIAS)
            images.append(self.__trans(img))
        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            pre_conf, pre_off = self.o_net(images)
        confidence = pre_conf.cpu()
        offset = pre_off.cpu()

        boxes = []
        indexs, _ = np.where(confidence > 0.9)

        for index in indexs:
            box = r_box[index]
            _x1 = float(box[0])
            _y1 = float(box[1])
            _x2 = float(box[2])
            _y2 = float(box[3])

            w = _x2 - _x1
            h = _y2 - _y1

            x1 = _x1 + w * offset[index][0]
            y1 = _y1 + h * offset[index][1]
            x2 = _x2 + w * offset[index][2]
            y2 = _y2 + h * offset[index][3]

            boxes.append([x1, y1, x2, y2, confidence[index][0]])
        boxes = np.array(torch.tensor(boxes))
        return utils.nms2(boxes, 0.5, is_min=True, softnms=True)

    def __box(self, index, offset, confidence, scale, stride=2, side_len=12):

        _x1 = (index[:, 1].float() * stride) / scale
        _y1 = (index[:, 0].float() * stride) / scale
        _x2 = (index[:, 1].float() * stride + side_len) / scale
        _y2 = (index[:, 0].float() * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, index[:, 0], index[:, 1]]

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        boxes = torch.stack([x1, y1, x2, y2, confidence]).transpose(0, 1)

        # boxes = []
        # for i in range(x1.shape[0]):
        #     boxes.append([x1[i].item(), y1[i].item(), x2[i].item(), y2[i].item(), confidence[i].item()])

        return np.array(boxes)
