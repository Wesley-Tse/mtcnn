#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/25 20:29

import os
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision import datasets, utils, transforms
from torch.utils.data import DataLoader, Dataset


class Sample(Dataset):
    def __init__(self, path):
        super(Sample, self).__init__()
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, 'positive.txt')).readlines())
        self.dataset.extend(open(os.path.join(path, 'negative.txt')).readlines())
        self.dataset.extend(open(os.path.join(path, 'part.txt')).readlines())

    def __getitem__(self, index):
        strs = self.dataset[index].split()
        img_path = os.path.join(self.path, strs[0])
        confidence = torch.tensor([int(strs[1])])
        offset = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        # offset = torch.Tensor([list(map(float, strs[2:6]))])
        key_point = torch.tensor(
            [float(strs[6]), float(strs[7]), float(strs[8]), float(strs[9]), float(strs[10]), float(strs[11]),
             float(strs[12]), float(strs[13]), float(strs[14]), float(strs[15])])

        image = self.__transform(Image.open(img_path))

        return image, confidence.float(), offset, key_point

    def __transform(self, x):
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
        ])
        return transform(x)

    def __len__(self):
        return len(self.dataset)
