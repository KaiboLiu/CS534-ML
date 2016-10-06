# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 16:04:36 2016

@author: Kaibo Liu
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

import GradientDecent1 as gd1

def HW1_part_1(trainX,trainY,lr=1,grad_epsilon=0.001,max_iterations=100000)

    learning_rate  = [0.001, 0.01, 0.1, 1, 10, 100]
#    max_iterations = 100000
#    grad_epsilon   = 0.001
#    lambda_reg     = [0, 0.0001, 0.001, 0.01, 1, 10, 100]

    for lr in learning_rate:
        [step,weight_hist, lossCont] = gd1.GradientDescent(trainX,trainY,lr,grad_epsilon,max_iterations)
        plt.plot(range(1,step+2),lossCont,'r--')
    plt.show()
    bestloss = np.min(lossCont)
    lr = 1
    return lr,bestloss,weight,lossCont
