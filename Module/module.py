#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Wesley
# @Time    : 2020.11.22 17:58

import torch
from torch import nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.convolution = nn.Sequential(
            # N * 3 * 12 * 12
            nn.Conv2d(3, 10, 3, 1),
            nn.PReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, 2),
            # N * 10 * 5 * 5
            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),
            nn.BatchNorm2d(16),
            # N * 16 * 3 * 3
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            # N * 32 * 1 * 1
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.off = nn.Conv2d(32, 4, 1, 1, bias=True)

    def forward(self, x):
        y = self.convolution(x)
        confidence = self.confidence(y)
        offset = self.off(y)

        return confidence, offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.convolution = nn.Sequential(
            # N * 3 * 24 * 24
            nn.Conv2d(3, 28, 3, 1, 1),
            nn.PReLU(),
            nn.BatchNorm2d(28),
            nn.MaxPool2d(3, 2),
            # N * 28 * 11 * 11
            nn.Conv2d(28, 48, 3, 1),
            nn.PReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(3, 2),
            # N * 48 * 4 * 4
            nn.Conv2d(48, 64, 2, 1),
            nn.PReLU(),
            nn.BatchNorm2d(64)
            # N * 64 * 3 * 3

        )
        self.layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128)
        )

        self.confidence = nn.Sequential(
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid()
        )
        self.off = nn.Linear(128, 4, bias=True)

    def forward(self, x):
        con_out = self.convolution(x)
        lin_out = self.layer(con_out.reshape(x.shape[0], -1))
        confidence = self.confidence(lin_out)
        offset = self.off(lin_out)

        return confidence, offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.convolution = nn.Sequential(
            # N * 3 * 48 * 48
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, 2),
            # N * 32 * 23 * 23
            nn.Conv2d(32, 64, 3, 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 2),
            # N * 64 * 10 * 10
            nn.Conv2d(64, 64, 3, 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            # N * 64 * 4 * 4
            nn.Conv2d(64, 128, 2, 1),
            nn.PReLU(),
            nn.BatchNorm2d(128),
            # N * 128 * 3 * 3
        )
        self.layer = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256)
        )
        self.confidence = nn.Sequential(
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid()
        )
        self.off = nn.Linear(256, 4, bias=True)

    def forward(self, x):
        con_out = self.convolution(x)
        lin_out = self.layer(con_out.reshape(x.shape[0], -1))
        confidence = self.confidence(lin_out)
        offset = self.off(lin_out)

        return confidence, offset


if __name__ == '__main__':
    p_net = PNet()
    r_net = RNet()
    o_net = ONet()

    img1 = torch.zeros([10, 3, 12, 12])
    img2 = torch.zeros([10, 3, 24, 24])
    img3 = torch.zeros([10, 3, 48, 48])

    c1, off1 = p_net(img1)
    print(c1.shape, off1.shape)
    c2, off2 = r_net(img2)
    print(c2.shape, off2.shape)
    c3, off3 = o_net(img3)
    print(c3.shape, off3.shape)
