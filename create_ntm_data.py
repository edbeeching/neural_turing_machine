#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:29:41 2018

@author: edward


    Generate example training sequences for the NTM

"""
import numpy as np
import matplotlib.pyplot as plt


def gen_sequence(num_bits=8, sequence_length=20):
    sequence = np.random.binomial(1, 0.5, (sequence_length, num_bits+1))
    sequence[:,num_bits] = 1.0
    

    
    return sequence.astype(np.float32)


if __name__ == '__main__':
    
    seq = gen_sequence()
    
    plt.imshow(seq)