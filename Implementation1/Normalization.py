# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 22:05:52 2016

@author: Kaibo Liu
"""
'''
Normalize all train data or normalize data in every data sample?
!!!!we should normalize every xi in all examples!!!Because x10 and x1d may have different unit,
but x10 and xN0 are in same unit. In conclusion, we should normalize the column vector of train data
the [0,1] normalization (also known as min-max) and the z-score normalization are two of the most widely used.
'''
import numpy as np

def Normalization(trainX,testX):
    smp_num, dim_num  = np.shape(trainX) #sample_num*dimension_num
    XzscoreNorm, XzscoreNormTest = trainX.copy(), testX.copy()
#    XminmaxNorm, XminmaxNormTest = trainX.copy(), testX.copy()
    for j in range(1,dim_num):
        X_this_dim = trainX[:,j]
        Xmean = np.mean(X_this_dim)
        Xstd  = np.std(X_this_dim)
        XzscoreNorm[:,j] = (trainX[:,j]-Xmean)/Xstd
        XzscoreNormTest[:,j]  = (testX[:,j]-Xmean)/Xstd
        '''
        Xmax = np.max(X_this_dim)
        Xmin = np.min(X_this_dim)
        XminmaxNorm[:,j] = (XminmaxNorm[:,j] - Xmean)/(Xmax-Xmin)
        XminmaxNormTest[:,j] = (XminmaxNormTest[:,j] - Xmean)(Xmax-Xmin)
        '''
        '''
        In python3 a/b is always a float, even if a and b are integeer
        '''
    return XzscoreNorm,XzscoreNormTest
#    return XminmaxNorm,XminmaxNormTest
