#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

import LoadData as ld
import Normalization as nor
import GradientDecent1 as gd1
import GradientDecent as gd
import CrossValidation as cv

def Output2File(wght_hist, lossCont, lr, lmd):
    #    np.savetxt("predict.csv",  predictCont,   delimiter = ',')
    np.savetxt("weght_lmd"+str(lmd)+"lr"+str(lr)+".csv",    wght_hist,   delimiter = ',')
    np.savetxt("loss_lmd"+str(lmd)+"lr"+str(lr)+".csv",     lossCont,      delimiter = ',')


def run_part2(trainX, trainY, testX, testY, lmd_reg, lr, eps, max_iter):
	final_loss     = []
	for lmd in lmd_reg:
		[weight_hist, lossCont, predictCont] = gd.GradientDescent(trainX, trainY, lr, eps, max_iter, lmd)
        	loss = lossCont[-1] #record the final loss.
        	final_loss.append(loss)

        print "for diff lambda, their final loss are:\n", final_loss
     	# print("Loss convergence history is: \n", lossCont)
        #  Output2File(weight_hist, lossCont, learning_rate, lmd)


def run_part3(trainX, trainY, testX, testY, lmd_reg, lr, eps, max_iter, k=1):
	valid_loss     = []
	for lmd in lmd_reg:		
		loss = cv.CrossValidation(trainX, trainY, lr, eps, max_iter, lmd, k)
		valid_loss.append(loss)
		print "loss_sum:", loss

	print "for diff lambda, their final validation loss are:", valid_loss	
	

def run():
	time.clock()
	t0 = float(time.clock())

	# load data from file, and do normalization on X.
	[trainX, trainY, testX, testY] = ld.LoadData()
	t1 = float(time.clock())
	print 'Loading data from File. using time %.4f s, \n' % (t1-t0)

	[trainX, testX] = nor.Normalization(trainX, testX)
	t2 = float(time.clock())
	print 'Normalization on train & test X. using time %.4f s, \n' % (t2-t1)

	# implementation assignments
	lr_reg   = [0.001, 0.01, 0.1, 1, 10, 100] #learning rate
	max_iter =  1000 # max iteration
	eps      =  0.001 # gradient comparing epsilon
	lmd_reg  = [0, 0.0001, 0.001, 0.01, 1, 10, 50] # regularization lambda

	# part 1, lamda = 0, different learning rate
	# [lr,bestloss,weight,lossCont] = HW1_part_1(trainX,trainY) #default lr,grad_epsilon and max_iterations
	t3 = float(time.clock())
	print 'Part 1, lamda = 0, changing lr, using time %.4f s, \n' %(t3-t2)

	# part2: fixed learning rate, different lamda
	best_lr = lr_reg[1]
	run_part2(trainX, trainY, testX, testY, lmd_reg, best_lr, eps, max_iter)
	t4 = float(time.clock())
	print 'Part 2, lamda = 0, changing lr, using time %.4f s, \n' %(t4-t3)

    	# part3: fixed lr, using 10-fold cross-validation
	# split training data into k parts
	k = 10
	run_part3(trainX, trainY, testX, testY, lmd_reg, best_lr, eps, max_iter, k)	

	
    # figure show.
'''
    idx = []
    for i in range(1, 51):
        idx.append(i)

    print idx

    plt.figure(1)
    plt.title("Accuracy for Given lr in Gaussian Descent")
    plt.plot(idx, lossCont)
    plt.show()

'''


if __name__ == '__main__':
    run()

