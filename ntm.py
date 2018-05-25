#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:49:19 2018

@author: edward

A simple NTM with only one read and write head

"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class NeuralTuringMachine(nn.Module):
    
    def __init__(self, batch_size, input_size, output_size, controller_size, memory_N, memory_M):
        super(NeuralTuringMachine, self).__init__()

        self.memory_N = memory_N
        self.memory_M = memory_M
        self.batch_size = batch_size
        self.controller_size=  controller_size
        # memory, N vectors of length M        
        self.register_buffer('initial_memory', Variable(torch.randn(memory_N, memory_M))*0.05)
        self.register_buffer('initial_read_bias', Variable(torch.randn(1, memory_M))*0.05)
        # TODO: Initialize the weights of the propotionally to size of memory

        self.controller = nn.LSTMCell(input_size + memory_M, controller_size)
        # TODO learn LSTM hidden state - DONE
        
        self.lstm_h_state_init = nn.Parameter(torch.ones(1, self.controller_size) * 0.05)
        self.lstm_c_state_init = nn.Parameter(torch.ones(1, self.controller_size) * 0.05)


        read_output_size = memory_M + 1 + 1 + 3 + 1
        write_output_size = memory_M + 1 + 1 + 3 + 1 + memory_M + memory_M
               
        self.fc_read_head = nn.Linear(controller_size, read_output_size)
        self.fc_write_head = nn.Linear(controller_size, write_output_size)        
        self.fc_output = nn.Linear(controller_size + memory_M, output_size)
        
        self.init_weights()
        
        self.reset()

    def init_weights(self):
        nn.init.orthogonal(self.fc_read_head.weight)
        nn.init.orthogonal(self.fc_write_head.weight)
        nn.init.orthogonal(self.fc_output.weight)
        nn.init.orthogonal(self.controller.weight_ih)
        nn.init.orthogonal(self.controller.weight_hh)
        self.controller.bias_ih = nn.Parameter(torch.randn(self.controller_size*4)*0.01)
        self.controller.bias_hh = nn.Parameter(torch.randn(self.controller_size*4)*0.01)
        

    def reset(self):
        self.memory = self.initial_memory.clone().repeat(self.batch_size, 1, 1)
        
        # create the hidden read states and write states
        #lstm_h_state = self.lstm_h_state_init.clone().repeat(self.batch_size, 1)
        
        #lstm_c_state = self.lstm_c_state_init.clone().repeat(self.batch_size, 1)
        lstm_h_state = Variable(torch.ones(self.batch_size, self.controller_size))
        lstm_c_state = Variable(torch.ones(self.batch_size, self.controller_size))
        
        
        
        self.controller_state = (lstm_h_state, lstm_c_state)
        self.read_state = Variable(torch.zeros(self.batch_size, self.memory_N))
        self.write_state = Variable(torch.zeros(self.batch_size, self.memory_N))
        self.previous_read = self.initial_read_bias.clone().repeat(self.batch_size, 1)
        #print('###### RESET #####')
        #print(self.controller.)
                
    def read_memory(self, weight_vector):

        assert weight_vector.size(1) == self.memory.size(1)
        assert weight_vector.size(0) == self.memory.size(0)
        
        return torch.sum(torch.matmul(weight_vector, self.memory), dim=1)
    
    def write_memory(self, weight_vector, erase_vector, add_vector):
        
        weight_erase = torch.matmul(weight_vector.unsqueeze(2), erase_vector.unsqueeze(1))
        
        self.memory = (self.memory * (1-weight_erase) 
                        + torch.matmul(weight_vector.unsqueeze(2), add_vector.unsqueeze(1)))
        

    def context(self, key_vector, beta_vector):
        # section 3.3.1 focusing by content
        # Key vector of length M
        # beta weighting scalar of length to change to precision of the focus
        
        cosine_similarity = F.cosine_similarity(self.memory + 1e-8, key_vector.unsqueeze(1) + 1e-8, dim=2)
        beta_similarity = beta_vector * cosine_similarity
        scores = F.softmax(beta_similarity, dim=1)
        
        return scores
    
    def interpolation(self, w_prev, w_current, g_scalar):
        # section 3.3.2 interpolation of previous weight vector and current weight vector
        return g_scalar * w_current + (1.0 - g_scalar) * w_prev
        
    
    def conv_shift(self, w_interp, shift):
        #section 3.3.2 circular convoluation
        
        w_pad = torch.cat([w_interp[:,-1:], w_interp, w_interp[:, :1]], dim=1).unsqueeze(0)        
        w_shift = F.conv1d(w_pad, shift.unsqueeze(1), groups=self.batch_size) # separable convolutions

        return w_shift.squeeze(0)       
    
    def sharpen(self, w_shift, sharp):
        #sections 3.3.2
        
        power = w_shift**sharp
        power_sum = torch.sum(power, dim=1) + 1e-8
        return power / power_sum
        
        
    def read_operation(self, x, w_prev):
        # get fc output, get beta etc, compute new weight vector and read 
        
        output = self.fc_read_head(x)
        # compute the head outputs
        key = output[:, :self.memory_M]
        beta = F.softplus(output[:, self.memory_M:self.memory_M+1]) 
        g_scalar = F.sigmoid(output[:, self.memory_M+1:self.memory_M+2]) 
        shift = F.softmax(output[:, self.memory_M+2:self.memory_M+5], dim=1)
        gamma = 1 + F.softplus(output[:, self.memory_M+5:self.memory_M+6])
        
        #context
        w_context = self.context(key, beta)
        #interp
        w_interp = self.interpolation(w_context, w_prev, g_scalar)
        #shift
        w_shift = self.conv_shift(w_interp, shift)
        #sharpen
        w_sharpen = self.sharpen(w_shift, gamma)
        #read memory
        read = self.read_memory(w_sharpen)
        return read, w_sharpen
    
    def write_operation(self, x, w_prev):
        
        output = self.fc_write_head(x)     
        # compute the head outputs
        key = output[:, :self.memory_M]
        beta = F.softplus(output[:, self.memory_M:self.memory_M+1]) 
        g_scalar = F.sigmoid(output[:, self.memory_M+1 : self.memory_M+2]) 
        shift = F.softmax(output[:, self.memory_M+2 : self.memory_M+5], dim=1)
        gamma = 1 + F.softplus(output[:, self.memory_M+5 : self.memory_M+6])
        erase = F.sigmoid(output[:, self.memory_M+6 : 2*self.memory_M+6]) 
        add = output[:, 2*self.memory_M+6 : 3*self.memory_M+6] # TODO non linearity

        #context
        w_context = self.context(key, beta)
        #interp
        w_interp = self.interpolation(w_context, w_prev, g_scalar)
        #shift
        w_shift = self.conv_shift(w_interp, shift)
        #sharpen
        w_sharpen = self.sharpen(w_shift, gamma)        
#        print(output, add)
        self.write_memory(w_sharpen, erase, add)
        
        return w_sharpen
        
    def forward(self, x):
        # updated the controller state
#        for name, param in self.named_parameters():
#            if param.requires_grad:
#                print(name)
#                print(param)
#                print(param.sum())
        
#        print('#'*80)
#        print('#'*80)
        #print(self.memory)
        controller_input = torch.cat([x, self.previous_read], dim=1)
        #print('##### INPUT #####')
        #print(controller_input)
        #print('##### STATE #####')
        #print(self.controller_state)
        controller_output, controller_cell = self.controller(controller_input, self.controller_state)
        #print('##### CONT OUTPUT #####')
        #print(controller_output)
        self.controller_state = (controller_output, controller_cell)
        read_output, self.read_state = self.read_operation(controller_output, self.read_state)
        self.write_state = self.write_operation(controller_output, self.write_state)
                
        output = F.sigmoid(self.fc_output(torch.cat([controller_output, read_output], dim=1)))
        #print('#'*80)
        #print('#'*80)
        return output
        

if __name__ == '__main__':
    ntm = NeuralTuringMachine(16, 8, 7, 32,  16, 12)
    
    
    output = ntm(Variable(torch.randn(16,8)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    