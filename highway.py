#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch

### YOUR CODE HERE for part 1h
class Highway(torch.nn.Module):
    """This module take in tensor:(batch_size,input), and output: (batch_size,input)"""
    def __init__(self,in_size,out_size):
        super(Highway, self).__init__()
        self.W_proj=torch.nn.Linear(in_size,out_size,bias=True)
        self.W_gate=torch.nn.Linear(in_size,out_size,bias=True)
        self.relu=torch.nn.functional.relu
        self.sigmod=torch.sigmoid

    def forward(self,x):
        x_gate=self.sigmod(self.W_gate(x))
        x_proj=self.relu(self.W_proj(x))
        return x_proj*x_gate + (1-x_gate)*x


### END YOUR CODE

