#-*- coding:utf8 -*-
#---- 20161002, SEOE, k-fold cross validation for Machine Learning Assignment

import numpy as np
#from sklearn.model_selection import KFold
import random

import GradientDecent as gd

def CrossValidation(x, y, lr, ep, max_iter, lmd, k):
	SSE = []
	#kf = KFold(n_splits=k, shuffle=True)
	smp_num, dim_num  = x.shape
        test_num = smp_num/k
        random_index = random.sample(xrange(0,smp_num), smp_num)

	#for train_index, test_index in kf.split(x):
	for i in range(k):
		test_index = random_index[:test_num]
                train_index = random_index[test_num:]

		# train on k-1 parts
		[weight_hist, lossCont] = gd.GradientDescent(np.take(x, train_index,axis=0), np.take(y,train_index,axis=0), lr, ep, max_iter, lmd)

		# test on the validation set and measure its SSE
		loss = gd.LossFunctions(np.take(x, test_index,axis=0), np.take(y,test_index,axis=0), weight_hist[-1], lmd)
		SSE.append(loss)		
		random_index = random_index[test_num:]+random_index[:test_num]

	return np.sum(SSE)
