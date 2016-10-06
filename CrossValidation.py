#-*- coding:utf8 -*-
#---- 20161002, SEOE, k-fold cross validation for Machine Learning Assignment

import numpy as np
from sklearn.model_selection import KFold

import GradientDecent as gd

def CrossValidation(x, y, lr, ep, max_iter, lmd, k):
	SSE = []
	kf = KFold(n_splits=k)

	i = 0
	for train_index, test_index in kf.split(x):
		print "lmd:", lmd, "k:", i
		# print("TRAIN:", train_index, "TEST:", test_index)

		# train on k-1 parts
		[weight_hist, lossCont, predictCont] = gd.GradientDescent(np.take(x, train_index,axis=0), np.take(y,train_index,axis=0), lr, ep, max_iter, lmd)

		# test on the validation set and measure its SSE
		loss = gd.LossFunctions(np.take(x, test_index,axis=0), np.take(y,test_index,axis=0), weight_hist[-1], lmd)
		SSE.append(loss)				
		i += 1	

	return np.sum(SSE)
