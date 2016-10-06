# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 22:05:52 2016

@author: Kaibo Liu
"""
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
    print trainX,trainY
    learning_rate  = [0.001,0.003,0.01,0.03,0.04,0.05]#,0.06,0.1]
#    lambda_reg     = [0, 0.0001, 0.001, 0.01, 1, 10, 100]
    color = 0
    plt.figure(1)
    for lr in learning_rate:
        [step,wght_hist, lossCont, predictCont] = gd1.GradientDescent(trainX,trainY,lr,grad_epsilon,max_iterations,0,50)
        plt.figure(1)
#        plt.plot(range(1,step+2),lossCont, get_colour(color),label="learning_rate %s" % str(lr))
        plt.plot(lossCont, get_colour(color),label="learning_rate %s" % str(lr))
 
        color += 1
#        title("learning_rate %s" % str(lr))
        plt.xlim(0, 400)
        plt.ylim(0, 60000)
  #      plt.ylim(0, 1e100)
        xlabel('iteration')
        ylabel('loss J(w)')  
        plt.legend(loc='upper right')
        grid(True)  
        savefig("Part1_converged Loss with learning rate_scaled")

        #    savefig("Part1_Loss-iter with learning_rate")  
    plt.show()
    
    bestloss = np.min(lossCont)
    lr = 1
    return lr,bestloss,wght_hist,lossCont

def run():
    # load data from file.
    time.clock()
    t0 = float(time.clock())
    [trainX, trainY, testX, testY] = ld.LoadData()
    tl = float(time.clock())    
    # part2: fixed learning rate, different lamda
    [trainX, testX] = nor.Normalization(trainX, testX)
#    learning_rate  = [0.001,0.03,0.1,0.3,1,3,10,30,100]

    '''part 1'''
    [lr,bestloss,weight,lossCont] = HW1_part_1(trainX,trainY) #default lr,grad_epsilon and max_iterations

    '''

    print 'Loading time: %.4f s' % (tl-t0)
    print 'Computing time:%.4f s' % (float(time.clock())-t0)



if __name__ == '__main__':
    run()

