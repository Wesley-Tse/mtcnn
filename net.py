#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Wesley
# @Time    : 2020.11.22 17:58

import torch
from torch import nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

    def forward(self, x):
        pass


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

    def forward(self, x):
        pass


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    p_net = PNet()
    r_net = RNet()
    o_net = ONet()

    img1 = torch.zeros([10, 3, 12, 12])
    img2 = torch.zeros([10, 3, 24, 24])
    img3 = torch.zeros([10, 3, 48, 48])

    c1, off1 = p_net(img1)
    c2, off2 = p_net(img2)
    c3, off3 = p_net(img3)

    print(c1, off1)
    print(c2, off2)
    print(c3, off3)
