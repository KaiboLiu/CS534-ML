"""
Created on Fri Nov 19 23:01:07 2016

@author: Kaibo Liu
"""
# for machine learning assignment4


#!/usr/bin/env python
import os
import numpy as np
from numpy import *  
import matplotlib.pyplot as plt
from scipy.stats import mode 
import time
import random

import LoadData   as ld

# calculate Euclidean distance, or square(error)
def Euclidean(x1, x2):  
    return sqrt(sum(power(x2 - x1, 2)))  
    #return sum(power(x2 - x1, 2))

# initiate k centers with random samples  
def initCentroids(Data, k):  
    n_data, dim = Data.shape  
    centers = random.sample(Data,k)   #randomly choose k samples as k initial centers, or: centroids = Data[np.random.choice(n_data,k,replace=False)]
    return np.array(centers)

# k-means cluster 
def kmeans(Data,k):
    n_data = len(Data)
    clusterInfo = np.zeros((n_data,2))   #first column is cluster # of this sample, second column is the dist^2(error) to its center
    clusterChanged = True

    centers = initCentroids(Data, k)  #initial centers by random choice

    #start interation
    while clusterChanged:
        clusterChanged = False
        #step 1: find nearest cluster center for each of the n data point
        for i in range(n_data):
            minDist = 1e20
            for j in range(k):  #search each center to assign the nearest center for this point
                dist = Euclidean(Data[i,:], centers[j,:])
                if dist < minDist:
                    minDist = dist
                    newCenter = j
            if clusterInfo[i,0] != newCenter:
                clusterChanged = True
                clusterInfo[i,:] = newCenter,minDist**2

        #step 2: update the centers
        for j in range(k):
            members = Data[np.where(clusterInfo[:,0]==j)[0]]  #filter the lines whose assigned cluster number is j
            centers[j,:] = np.mean(members,axis = 0)

    SSE = sum(clusterInfo[:,1])
    print 'SSE %.4f,' %(SSE),
    for j in range(k):
        print 'cluster%d:%d,' %(j,clusterInfo[:,0].tolist().count(j)),

    return clusterInfo, SSE

        
def get_colour(color):
    '''
    b---blue   c---cyan  g---green    k----black
    m---magenta r---red  w---white    y----yellow'''
    color_set = ['r','b','m','g','c','k','y']
    return color_set[color % 7]

def RunMain():
    print '\n************ Welcome to the World of Demension Reduction & Clustering! ***********'
    time.clock()
    t0 = t00 = float(time.clock())
    # # load data, and save data.
    [Data, Label] = ld.LoadData()
    t1 = float(time.clock())
    print '[done] Loading data File. using time %.4f s.\n' % (t1-t0)

    saveDir    = "./walking-Result/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    #******************Part 1***************************
    #Kmeans clustering
    print '******Part 1******'
    t01 = float(time.clock())
    n_data = len(Data)
    randomRuns = 10
    k = 2
    minSSE, accuracy = 1e20 , 0
    for i in range(randomRuns):
        t0 = t1
        print '#%d kmeans:' %(i+1),
        clusterInfo, SSE = kmeans(Data,k)
        if SSE < minSSE:
            minSSE = SSE
            bestCluster = clusterInfo
        t1 = float(time.clock())
        print 'using time %.4f s.' % (t1-t0)


    #measure purity
    labelPredict = np.zeros(n_data)
    for i in range(k):
        members = np.where(bestCluster[:,0] == i)[0]
        majorityClass = mode(Label[members])[0][0]
        labelPredict[members] = majorityClass
    error = sum(abs(labelPredict-Label))
    purity = 100 - 100.0*float(error)/n_data
    t1 = float(time.clock())
    print '[done] purity with best SSE is %.3f%%, time for part 1 is %.4f s. \n' % (purity,t1-t01)
    



    #******************Part 2***************************
    print '******Part 2******'
 




    t1 = float(time.clock())
    print '[done] total runtime is %.4f s. \n' % (t1-t00)


if __name__ == "__main__":
    RunMain()
