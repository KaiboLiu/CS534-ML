# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 22:05:52 2016
@author: Kaibo Liu
Gradient Decent for Machine Learning Assignment
"""

import matplotlib.pyplot as plt
import pylab
from pylab import *
import numpy as np  #vector & matrix
import math
import random       # random
#import sklearn      #scikit-learn, machine learning in Python
#from sklearn.datasets.samples_generator import make_regression
#import pylab #to add label?
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

def GradientDescent(x, y, lr, ep, max_iter, lmd = 0, rcdNum = 50):
	# loop control variables
    converged   = False
#    iter        = 0
    rcdStep     = max_iter/rcdNum;
    # prepare output data container
    lossCont    = []
    predictCont = []
    wght_hist   = []
    
    	# initial weight
    smp_num, dim_num  = x.shape  # sample numbers, and feature dimensions,n*d	
    wght              = np.zeros(dim_num)  #d*1

    # Iteration loop, converge by gradient decend
    for i in range (max_iter):
        predict = np.dot(x, wght)  # predict value
        error   = predict - y
        loss    = np.sum(error**2) + lmd * np.sum((wght)**2)
        '''		
        if i%rcdStep == 0:
            lossCont.append(loss)
            predictCont.append(predict)
'''
        lossCont.append(loss)
        predictCont.append(predict)

        if loss > 1.1e100:
            lossCont[-1] = 1.1e100;
            print "Not Converged, lrearning rate %s, #iter %d, loss MAX" % (str(lr),i)
            break
        
        
        #for each training sample, calc its gradient 2*lmd*wght
        grad = (np.dot(x.T,error)*2)/smp_num + 2*lmd*wght
        wght = wght - lr * grad
        wght_hist.append(wght)
        if np.sqrt(np.sum(grad**2)) < ep:
            print "Converged, lrearning rate %s, #iter %d, loss %.4f" % (str(lr),i,loss)
            break

            #np.sum(abs(grad)) < ep
           
#    print 'grad',grad
#    print i,np.sqrt(np.sum(grad**2)),loss
#    plt.figure(2)
#    plt.plot(g)
#    show()
    return [wght_hist, lossCont, predictCont]

