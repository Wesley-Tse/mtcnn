#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020/11/25 20:28

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.get_data import GetSample
from Module.module import PNet

class Trainer:
    def __init__(self, net, params, dataset, device, batch_size, epoch):
        self.net = net.to(device)
        self.device = device
        self.params = params
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = epoch
        self.optimizer = torch.optim.Adam(self.params(), lr=0.001)
        self.loss_con = nn.BCEWithLogitsLoss()
        self.loss_off = nn.MSELoss()
        if os.path.exists(self.params):
            net.load_state_dict = torch.load(self.params)

    def train(self):
        data = DataLoader(dataset=GetSample(self.dataset), batch_size=self.batch_size, shuffle=True, drop_last=True)
        epoch = 0
        while epoch < self.epoch:
            print('--------------epoch{}/{}--------------'.format(epoch, self.epoch))
            for i, (img, confidence, offset, key_point) in enumerate(data):
                img = img.to(self.device)
                confidence = confidence.to(self.device)
                offset = offset.to(self.device)
                key_point = key_point.to(self.device)
                # 网络输出
                pre_conf, pre_off, pre_key = self.net(img)
                # 置信度loss
                con_mask = torch.lt(confidence, 2)
                confidence_ = torch.masked_select(confidence, con_mask)
                pre_conf_ = torch.masked_select(pre_conf, con_mask)
                loss_con = self.loss_con(pre_conf_, confidence_)
                # 偏移量loss
                off_mask = torch.gt(confidence, 0)
                offset_ = torch.masked_select(offset, off_mask)
                pre_off_ = torch.masked_select(pre_off, off_mask)
                loss_off = self.loss_off(pre_off, offset_)
                # 计算loss，更新参数
                loss = loss_con + loss_off
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 50 == 0:
                    print('loss', loss)
                    torch.save(self.net.state_dict() , self.params)
                    print('params save success!')

            epoch += 1

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = r'D:\PycharmProjects\Datasets\datasets3\48'
    params = './config/test.pt'
    net = PNet()
    trainer = Trainer(net, params, dataset, device, 512, 10)
    trainer.train()