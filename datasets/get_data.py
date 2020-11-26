#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/25 20:29

import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class GetSample(Dataset):
    def __init__(self, path):
        super(GetSample, self).__init__()
        self.path = path
        self.dataset = []
        with open(os.path.join(path, 'positive.txt')) as positive:
            self.dataset.extend(positive.readlines())
        with open(os.path.join(path, 'negative.txt')) as negative:
            self.dataset.extend(negative.readlines())
        with open(os.path.join(path, 'part.txt')) as part:
            self.dataset.extend(part.readlines())

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
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
        ])
        return transform(x)

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    path = r'D:\Datasets\12'
    data = DataLoader(dataset=GetSample(path), batch_size=10, shuffle=True, drop_last=True)
    for i, (img, confidence, offset, key_point) in enumerate(data):
        print(confidence.shape)
        print(offset.shape)
        print(key_point.shape)