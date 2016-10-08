#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab
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

def get_colour(color):
    '''
    b---blue   c---cyan  g---green    k----black
    m---magenta r---red  w---white    y----yellow
'''
    color_set = ['r--','b--','m--','g--','c--','k--','y--']
    return color_set[color % 7]


def HW1_part_1(trainX,trainY,grad_epsilon=0.01,max_iterations=100000):
    learning_rate  = [0.01,0.03,0.05,0.06]
    color = 0
    plt.figure(1)
    for lr in learning_rate:
        [wght_hist, lossCont, predictCont] = gd.GradientDescent(trainX,trainY,lr,grad_epsilon,max_iterations,0,50)
        plt.figure(1)
#        plt.plot(range(1,step+2),lossCont, get_colour(color),label="learning_rate %s" % str(lr))
        plt.plot(lossCont, get_colour(color),label="learning_rate %s" % str(lr))
        color += 1
#        title("learning_rate %s" % str(lr))
        plt.xlim(0, 2000)
        plt.ylim(200, 2000)
        plt.xlabel('iteration')
        plt.ylabel('loss J(w)')  
        plt.legend(loc='upper right')
        plt.grid(True)  
   # pdb.set_trace()
    plt.savefig("Part1_converged Loss with learning rate_scaled-Para")
    print 'para:MaxIter=%d,Normalization=ZscoreNorm,L1-Norm<epsilon=%s,LossThreshold=1e100,grad/n' % (max_iterations,str(grad_epsilon))
#        savefig("Part1_unconverged Loss-Para")

        #    savefig("Part1_Loss-iter with learning_rate")  
   # plt.show()
    
   # pdb.set_trace()

    bestloss_Part1 = np.min(lossCont)
    lr = 0.05 #According to the convergence and figure results above
    print 'According to the convergence and figure results from Part1, we decide learning rate as %s' % str(lr)
    return lr

def run_part2(trainX, trainY, testX, testY, lmd_reg, lr, eps, max_iter):
	#pdb.set_trace()
	finalWeight       = []
	trainLoss    = []
	testLoss     = []
	for lmd in lmd_reg:
		# training
		[weight_hist, lossCont, predictCont] = gd.GradientDescent(trainX, trainY, lr, eps, max_iter, lmd)
		finalWeight.append(weight_hist[-1])
		trainLoss.append(lossCont[-1]) #record the final loss.
		#print "lambda = "+str(lmd)+" converge loss are: \n", lossCont
		#print "final weight is: \n", weight_hist[-1]

		# test with trained w.
		teloss = gd.LossFunctions(testX, testY, weight_hist[-1], lmd)
		testLoss.append(teloss)

	print "for diff lambda, traning loss are:\n", trainLoss
	print "                 test loss are: \n", testLoss


def run_part3(trainX, trainY, testX, testY, lr, eps, max_iter, lmd_reg, k=1):
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
	best_lr = HW1_part_1(trainX,trainY) #default lr,grad_epsilon and max_iterations
	# [lr,bestloss,weight,lossCont] = HW1_part_1(trainX,trainY) #default lr,grad_epsilon and max_iterations
	t3 = float(time.clock())
	print 'Part 1, lamda = 0, changing lr, using time %.4f s, \n' %(t3-t2)

	# part2: fixed learning rate, different lamda
	run_part2(trainX, trainY, testX, testY, lmd_reg, best_lr, eps, max_iter)
	t4 = float(time.clock())
	print 'Part 2, lamda = 0, changing lr, using time %.4f s, \n' %(t4-t3)

    	# part3: fixed lr, using 10-fold cross-validation
	# split training data into k parts
	k = 10
	run_part3(trainX, trainY, testX, testY, best_lr, eps, max_iter, lmd_reg, k)	

	
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

