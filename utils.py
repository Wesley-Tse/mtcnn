#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/25 20:29

import numpy as np


def Iou(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 计算相交框的坐标
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    # 相交面积
    inter = w * h

    if isMin:
        # 最小面积

        iou = inter / np.minimum(box_area, boxes_area)
    else:
        iou = inter / (box_area + boxes_area - inter)

    return iou


def Nms(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])
    boxes_sort = boxes[(-boxes[:, 4]).argsort()]
    # boxes_sort = boxes[boxes[:, 4].argsort()[::-1]]
    use_boxes = boxes_sort.copy()
    reserve_boxes = []
    while use_boxes.shape[0] > 1:
        first_box = use_boxes[0]
        remain_boxes = use_boxes[1:]
        reserve_boxes.append(first_box)

        iou = Iou(first_box, remain_boxes, isMin)

        index = np.where(iou < thresh)
        use_boxes = remain_boxes[index]
    if use_boxes.shape[0] == 1:
        reserve_boxes.append(use_boxes[0])

    reserve_boxes = np.stack(reserve_boxes)
    return reserve_boxes


def Expan_box(box):
    new_box = box.copy()
    if new_box.shape[0] == 0:
        return np.array([])
    w = new_box[:, 2] - new_box[:, 0]
    h = new_box[:, 3] - new_box[:, 1]

    max_side = np.maximum(w, h)

    center_x = new_box[:, 0] + w * 0.5
    center_y = new_box[:, 1] + h * 0.5

    new_box[:, 0] = center_x - max_side * 0.5
    new_box[:, 1] = center_y - max_side * 0.5
    new_box[:, 2] = new_box[:, 0] + max_side
    new_box[:, 3] = new_box[:, 1] + max_side

    return new_box
