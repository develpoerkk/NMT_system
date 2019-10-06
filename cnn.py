#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch

### YOUR CODE HERE for part 1i
class CNN(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5):
        super(CNN,self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.conv=torch.nn.Conv1d(in_channels,out_channels,kernel_size)
    def forward(self,x):
#        print(x.size())
        x_conv=self.conv(x)
#        print(self.conv.bias)
        return torch.max(torch.relu(x_conv),dim=-1)[0]


### END YOUR CODE

