#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:41:28 2018

@author: edward
"""
import numpy as np
import matplotlib.pyplot as plt
from ntm import NeuralTuringMachine
from create_ntm_data import gen_sequence

import torch
from torch.autograd import Variable
from torch import Tensor, nn


if __name__ == '__main__':

    BATCH_SIZE = 16
    INPUT_SIZE = 12
    OUTPUT_SIZE = 11
    CONTROLLER_SIZE = 32
    MEMORY_N = 32
    MEMORY_M = 16
    
    SEQUENCE_LENGTH = 16
    
    ntm = NeuralTuringMachine(BATCH_SIZE, INPUT_SIZE, 
                              OUTPUT_SIZE, CONTROLLER_SIZE,
                              MEMORY_M, MEMORY_N)
    
    training_data = gen_sequence(num_bits=INPUT_SIZE-1, sequence_length=SEQUENCE_LENGTH)
    
    loss_fn = nn.BCELoss()
    optimizer= torch.optim.RMSprop(ntm.parameters(),lr=5e-3, momentum=0.9)
    print('Printing layer names')
    for name, param in ntm.named_parameters():
        if param.requires_grad:
            print(name)

    for epoch in range(200):
        
        ntm.reset()
        for piece in training_data:
            data = Variable(Tensor(piece)).unsqueeze(0)
            ntm(data)
        output = []
        for piece in training_data:
            data = Variable(torch.zeros(BATCH_SIZE, INPUT_SIZE))
            out = ntm(data)            
            output.append(out)
            
        loss = loss_fn(torch.cat(output), Variable(Tensor(training_data[:, :-1])))
        
        if epoch % 10 == 0: 
            print(epoch, loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        
#        print('#'*80)
#        print('#'*80)
#        for name, param in ntm.named_parameters():
#            if param.requires_grad:
#                print(name)
#                print(param)
#                print(param.grad)
                #print(param.sum())
        
        #nn.utils.clip_grad_norm(ntm.parameters(), 10.0, norm_type=1)
        optimizer.step()
        
            
    
    
    plt.subplot(1,2,1)
    plt.imshow(training_data[:,:-1])   
    plt.subplot(1,2,2)
    plt.imshow(torch.cat(output).data.numpy())
    
    
    
    