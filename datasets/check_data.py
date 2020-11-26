#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/26 19:17

import os
from PIL import Image, ImageDraw
from matplotlib import pyplot

img_path = r'D:\CelebA_data\CelebA\Img\img_celeba.7z\img_celeba'
label_path = r'D:\CelebA_data\CelebA\Anno\list_bbox_celeba.txt'

with open(label_path, 'r', encoding='utf-8') as label:
    for i, line in enumerate(label.readlines()):
        if i < 2:
            continue
        strs = line.split()

        img_name = strs[0].strip()
        img_file = os.path.join(img_path, img_name)
        x1 = int(strs[1])
        y1 = int(strs[2])
        w = int(strs[3])
        h = int(strs[4])
        x2 = w + x1
        y2 = h + y1

        with Image.open(img_file) as img:
            img_draw = ImageDraw.Draw(img)
            img_draw.rectangle((x1, y1, x2, y2), outline='red', width=3)
            pyplot.clf()
            # pyplot.scatter(float(strs[5]), float(strs[6]), color='green', marker='.')
            # pyplot.scatter(float(strs[7]), float(strs[8]), color='green', marker='.')
            # pyplot.scatter(float(strs[9]), float(strs[10]), color='green', marker='.')
            # pyplot.scatter(float(strs[11]), float(strs[12]), color='green', marker='.')
            # pyplot.scatter(float(strs[13]), float(strs[14]), color='green', marker='.')

            pyplot.imshow(img)
            pyplot.pause(1)