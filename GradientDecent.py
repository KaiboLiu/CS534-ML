#-*- coding:utf8 -*-
#---- 20161001, YJL, Gradient Decent for Machine Learning Assignment

import numpy as np  #vector & matrix
import random       # random
import sklearn      #scikit-learn, machine learning in Python
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats

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
	iter        = 0
	rcdStep     = max_iter/rcdNum;

	# prepare output data container
	lossCont    = []
	predictCont = []
	wght_hist   = []

	# initial weight
	smp_num, dim_num  = x.shape  # sample numbers, and feature dimensions	
	wght              = np.random.random(dim_num)

	# Iteration loop, converge by gradient decend
	for i in range (max_iter):
		predict = np.dot(x, wght)  # predict value
		error   = predict - y
		loss    = np.sum(error**2) + lmd * np.sum((wght)**2)
		
		if i%rcdStep == 0:
			lossCont.append(loss)
			predictCont.append(predict)

		#for each training sample, calc its gradient
		grad = (np.dot(x.T,error)*2)/smp_num + 2*lmd*wght
		wght = wght - lr * grad
		wght_hist.append(wght)

		if grad.max() < ep:
			print "gradient convergence to a small value, stop optimization"
			break

	return [wght_hist, lossCont, predictCont]

def LossFunctions(x, y, w, lmd):
	predict = np.dot(x, w)  # predict value
	error   = predict - y
	return np.sum(error**2) + lmd * np.sum((w)**2)
		

