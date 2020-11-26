#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/25 20:29

import torch
from PIL import Image
from matplotlib import pyplot
from detect.detector import Detector
if __name__ == '__main__':

    image_path = r'E:\PyCharmProject\mtcnn\src\images\2.jpg'

    p_net_param = r'E:\PyCharmProject\mtcnn\config\p.pt'
    r_net_param = r'E:\PyCharmProject\mtcnn\config\r.pt'
    o_net_param = r'E:\PyCharmProject\mtcnn\config\o.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detector = Detector(p_net_param, r_net_param, o_net_param, device)

    with Image.open(image_path) as img:
        print(img.size)
        boxes = detector.detect(img)
        print(boxes)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            pyplot.gca().add_patch(
                pyplot.Rectangle((x1, y1), width=x2 - x1, height=y2 - y1, edgecolor='red', facecolor='none')
            )
            # pyplot.scatter(int(box[5]), int(box[6]), color='green', marker='.')
            # pyplot.scatter(int(box[7]), int(box[8]), color='green', marker='.')
            # pyplot.scatter(int(box[9]), int(box[10]), color='green', marker='.')
            # pyplot.scatter(int(box[11]), int(box[12]), color='green', marker='.')
            # pyplot.scatter(int(box[13]), int(box[14]), color='green', marker='.')

        pyplot.imshow(img)
        pyplot.show()