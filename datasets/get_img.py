#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/26 11:50

import os
import random
import numpy as np
import utils
from PIL import Image

if __name__ == '__main__':

    Image_path = r'D:\CelebA_data\CelebA\Img\img_align_celeba\img_align_celeba'
    Label_path = r'E:\PyCharmProject\mtcnn\datasets\label.txt'

    Save_path = r'D:\Datasets'

    for face_size in [12, 24, 48]:
        # 创建样本保存文件夹
        positive_sample_dir = os.path.join(Save_path, str(face_size), 'positive')
        negative_sample_dir = os.path.join(Save_path, str(face_size), 'negative')
        part_sample_dir = os.path.join(Save_path, str(face_size), 'part')

        for dir in [positive_sample_dir, negative_sample_dir, part_sample_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
        # 创建标签保存文件
        positive_sample_label = os.path.join(Save_path, str(face_size), 'positive.txt')
        negative_sample_label = os.path.join(Save_path, str(face_size), 'negative.txt')
        part_sample_label = os.path.join(Save_path, str(face_size), 'part.txt')

        # 样本计数器
        positive_count = 0
        negative_count = 0
        part_count = 0

        # 打开标签文件
        positive = open(positive_sample_label, 'w')
        negative = open(negative_sample_label, 'w')
        part = open(part_sample_label, 'w')

        origin_label = open(Label_path, 'r')

        for i, line in enumerate(origin_label.readlines()):
            if i < 2:
                continue
            if i == 1000:
                break
            strs = line.split()
            img_filename = strs[0].strip()
            img_path = os.path.join(Image_path, img_filename)

            with Image.open(img_path) as img:
                img_w, img_h = img.size
                x1 = int(strs[1])
                y1 = int(strs[2])
                box_w = int(strs[3])
                box_h = int(strs[4])
                x2 = x1 + box_w
                y2 = y1 + box_h

                # 关键点
                l_eye_x, l_eye_y = int(strs[5]), int(strs[6])
                r_eye_x, r_eye_y = int(strs[7]), int(strs[8])
                nose_x, nose_y = int(strs[9]), int(strs[10])
                l_mouth_x, l_mouth_y = int(strs[11]), int(strs[12])
                r_mouth_x, r_mouth_y = int(strs[13]), int(strs[14])

                # 处理过小框和错误框
                if min(box_w, box_h) < 60 or x1 < 0 or y1 < 0 or box_w < 0 or box_h < 0:
                    continue

                box = np.array([[x1, y1, x2, y2]])

                # 计算中心点
                c_x = x1 + box_w * 0.5
                c_y = y1 + box_h * 0.5

                # 生成样本
                positive_flag = 0
                part_flag = 0
                negative_flag = 0

                for _ in range(2):
                    # 给中心点一个随机偏移量
                    w_ = np.random.randint(int(-box_w * 0.2), int(box_w * 0.2))
                    h_ = np.random.randint(int(-box_h * 0.2), int(box_h * 0.2))

                    # 偏移后的中心点
                    new_c_x = c_x + w_
                    new_c_y = c_y + h_

                    # 生成随机正方形框
                    rand_side = random.randint(int(max(box_w, box_h) * 0.9), int(max(box_w, box_h)))

                    new_x1 = max(c_x - rand_side * 0.5, 0)
                    new_y1 = max(c_y - rand_side * 0.5, 0)
                    new_x2 = new_x1 + rand_side
                    new_y2 = new_y1 + rand_side

                    crop_box = np.array([new_x1, new_y1, new_x2, new_y2])

                    # 计算偏移量
                    off_x1 = (x1 - new_x1) / rand_side
                    off_y1 = (y1 - new_y1) / rand_side
                    off_x2 = (x2 - new_x2) / rand_side
                    off_y2 = (y2 - new_y2) / rand_side

                    off_l_eye_x = (l_eye_x - new_c_x) / rand_side
                    off_l_eye_y = (l_eye_y - new_c_y) / rand_side

                    off_r_eye_x = (r_eye_x - new_c_x) / rand_side
                    off_r_eye_y = (r_eye_y - new_c_y) / rand_side

                    off_nose_x = (nose_x - new_c_x) / rand_side
                    off_nose_y = (nose_y - new_c_y) / rand_side

                    off_l_mouth_x = (l_mouth_x - new_c_x) / rand_side
                    off_l_mouth_y = (l_mouth_y - new_c_y) / rand_side

                    off_r_mouth_x = (r_mouth_x - new_c_x) / rand_side
                    off_r_mouth_y = (r_mouth_y - new_c_y) / rand_side

                    # 将生成的框从原图扣出
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size))

                    # 计算生成框和原框的iou
                    iou = utils.iou(crop_box, box)[0]

                    if positive_flag < 1 and iou > 0.65:
                        positive.write(
                            'positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n'.format(
                                positive_count, 1, off_x1, off_y1, off_x2, off_y2, off_l_eye_x, off_l_eye_y,
                                off_r_eye_x, off_r_eye_y, off_nose_x, off_nose_y, off_l_mouth_x, off_l_mouth_y,
                                off_r_mouth_x, off_r_mouth_y))
                        positive.flush()
                        face_resize.save(os.path.join(positive_sample_dir, '{}.jpg'.format(positive_count)))
                        positive_count += 1
                        positive_flag += 1

                for _ in range(15):
                    # 随机生成框
                    side = random.randint(int(min(box_w, box_h) * 0.9), int(min(box_w, box_h) * 0.9))
                    # cen_x = random.randint(x1, x2)
                    # cen_y = random.randint(y1, y2)
                    cen_x = random.randint(int(strs[5]) - 5, int(strs[7]) + 5)
                    cen_y = random.randint(int(strs[6]) - 15, int(strs[12]) + 15)

                    idx = random.randint(0, 3)

                    cen_points = np.array([(cen_x, float(strs[6])), (cen_x, float(strs[9])), (float(strs[5]), cen_y),
                                           (float(strs[12]), cen_y)])
                    cen_point = cen_points[idx]

                    new_c_x = cen_point[0].item()
                    new_c_y = cen_point[1].item()

                    new_x1 = new_c_x - side * 0.5
                    new_y1 = new_c_y - side * 0.5
                    new_x2 = new_c_x + side * 0.5
                    new_y2 = new_c_y + side * 0.5

                    crop_box = np.array([new_x1, new_y1, new_x2, new_y2])

                    # 计算偏移量
                    off_x1 = (x1 - new_x1) / rand_side
                    off_y1 = (y1 - new_y1) / rand_side
                    off_x2 = (x2 - new_x2) / rand_side
                    off_y2 = (y2 - new_y2) / rand_side

                    off_l_eye_x = (l_eye_x - new_c_x) / rand_side
                    off_l_eye_y = (l_eye_y - new_c_y) / rand_side

                    off_r_eye_x = (r_eye_x - new_c_x) / rand_side
                    off_r_eye_y = (r_eye_y - new_c_y) / rand_side

                    off_nose_x = (nose_x - new_c_x) / rand_side
                    off_nose_y = (nose_y - new_c_y) / rand_side

                    off_l_mouth_x = (l_mouth_x - new_c_x) / rand_side
                    off_l_mouth_y = (l_mouth_y - new_c_y) / rand_side

                    off_r_mouth_x = (r_mouth_x - new_c_x) / rand_side
                    off_r_mouth_y = (r_mouth_y - new_c_y) / rand_side

                    # 将生成的框从原图扣出
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size))

                    # 计算生成框和原框的iou
                    iou = utils.iou(crop_box, box)[0]

                    if part_flag < 1 and iou < 0.6 and iou > 0.4:
                        part.write(
                            'part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n'.format(
                                part_count, 2, off_x1, off_y1, off_x2, off_y2, off_l_eye_x, off_l_eye_y,
                                off_r_eye_x, off_r_eye_y, off_nose_x, off_nose_y, off_l_mouth_x, off_l_mouth_y,
                                off_r_mouth_x, off_r_mouth_y))
                        part.flush()
                        face_resize.save(os.path.join(part_sample_dir, '{}.jpg'.format(part_count)))
                        part_count += 1
                        part_flag += 1

                    # 单独造负样本
                _box = box.copy()
                for _ in range(10):
                    side_len = face_size
                    x_ = random.randint(0, img_w - side_len)
                    y_ = random.randint(0, img_h - side_len)

                    _crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                    if negative_flag < 3 and max(utils.Iou(_crop_box, _box)) == 0:
                        face_crop = img.crop(_crop_box)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                        negative.write(
                            'negative/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n'.format(
                                negative_count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                        negative.flush()
                        face_resize.save(os.path.join(negative_sample_dir, '{}.jpg'.format(negative_count)))
                        negative_count += 1
                        negative_flag += 1

                print('{}--生成{}x{}样本'.format(img_filename, face_size, face_size))
        print(
            '{}x{}的样本共造了{}个正样本，{}个部分样本，{}个负样本'.format(face_size, face_size, positive_count, part_count, negative_count))
