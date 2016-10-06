# -*- coding: utf-8 -*-

#-*- coding:utf8 -*-
#---- 20161001, YJL, Gradient Decent for Machine Learning Assignment

import numpy as np  #vector & matrix
import math
import random       # random
import sklearn      #scikit-learn, machine learning in Python
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import matplotlib.pyplot as plt

#### Param: ###############
# function: error = sum(yi - w^T * xi)^2 + \lambda * ||w||^2
#x / y: feature & observator
# lr  : learning rate, weight_new = weight_old - lr * \partial(error)
# ep  : epsilon, used to stop convergence
# max_iter: maximum iteration number if non-convergence.
# lmd : regularized factor in regular SSE model, = 0 if normal SSE
#rcdNum: output record number.
###### Output: ###############
# weight w : model parameter.
###############################



def GradientDescent(x, y, lr, ep, max_iter,rcdNum = 100):
    # loop control variables
    converged   = False
#    iter        = 0
    rcdStep     = max_iter/rcdNum

    # prepare output data container
    lossCont    = []
    predictCont = []
    wght_hist   = []

    # initial weight
#    print(x)
#    print(x.shape)
    smp_num, dim_num  = x.shape  # sample numbers, and feature dimensions   
#    smp_num, dim_num  = np.shape(x)  # sample numbers, and feature dimensions    
    wght              = np.random.random(dim_num)
    # Iteration loop, converge by gradient decend
    for i in range (rcdStep):
        predict = np.dot(x, wght)  # predict value
#        print predict
        error   = predict - y
        loss    = np.sum(error**2)
        
#        if i < rcdStep:
        lossCont.append(loss)
#        predictCont.append(predict)

        #for each training sample, calc its gradient
#        grad = np.dot(x.T,error)*(2/smp_num)
        grad = np.dot(x.T,error)
        wght = wght - lr * grad
        wght_hist.append(wght)
#        print grad
#        print grad.shape
#        print np.dot(grad,grad)
#        print np.sqrt(np.dot(grad,grad))
#        if np.linalg.norm(grad,order=1) < ep:
        if np.sqrt(np.dot(grad,grad)) < ep:
            print "gradient convergence to a small value, stop optimization, #iter=%d, SSE=%.6f." % (i,lossCont[-1])
#            print "gradient convergence to a small value, stop optimization"
            break

    return [i,wght_hist, lossCont]
