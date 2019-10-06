# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:12:11 2019

@author: KK
"""

import cnn
import highway
import torch

c_param=torch.nn.Parameter(torch.rand(1,2,2))
c_input=torch.tensor([[[1,1],[1,1]]],dtype=torch.double)
CNN=cnn.CNN(2,1,2)
print(CNN(c_input))