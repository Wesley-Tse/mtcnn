#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/25 20:29

import numpy as np


def iou(box, boxes, isMin=False):
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

        out = inter / np.minimum(box_area, boxes_area)
    else:
        out = inter / (box_area + boxes_area - inter)

    return out


def nms(boxes, thresh=0.3, isMin=False):
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

        out = iou(first_box, remain_boxes, isMin)

        index = np.where(out < thresh)
        use_boxes = remain_boxes[index]
    if use_boxes.shape[0] == 1:
        reserve_boxes.append(use_boxes[0])

    reserve_boxes = np.stack(reserve_boxes)
    return reserve_boxes


def expand_box(box):
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

def nms2(boxes, thresh=0.3, is_min=False, softnms=False):
    if boxes.shape[0] == 0:
        return np.array([])
    _boxes = boxes[(-boxes[:, 4]).argsort()]    # 按置信度排序
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]
        score = b_boxes[:, 4]
        r_boxes.append(a_box)

        if softnms:
            score_thresh = 0.5
            # IOU>阈值的框 置信度衰减
            t_idx = np.where(iou(a_box, b_boxes, is_min) > thresh)
            score[t_idx] *= (1 - iou(a_box, b_boxes, is_min))[t_idx]
            # 删除分数<阈值的框
            _boxes = np.delete(b_boxes, np.where(score < score_thresh), axis=0)
        else:
            # 筛选IOU<阈值的框
            index = np.where(iou(a_box, b_boxes, is_min) < thresh)
            _boxes = b_boxes[index]

    # 剩余最后1个框 保留
    if _boxes.shape[0] == 1:
        r_boxes.append(_boxes[0])

    # 把list组装成矩阵
    return np.stack(r_boxes)
