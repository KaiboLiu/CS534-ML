"""
Created on Fri Nov 19 23:01:07 2016

@author: Kaibo Liu
"""
# for machine learning assignment4


#!/usr/bin/env python
import os
import numpy as np
from numpy import *
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import mode
import time
import random
import pdb

import LoadData   as ld

# calculate Euclidean distance, or square(error)
def Euclidean(x1, x2):
    return sqrt(sum(power(x2 - x1, 2)))
    #return sum(power(x2 - x1, 2))

# initiate k centers with random samples
def initCentroids(Data, k):
    n_data  = len(Data)
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
            minDist = 1e100
            for j in range(k):  #search each center to assign the nearest center for this point
                dist = Euclidean(Data[i], centers[j])
                if dist < minDist:
                    minDist = dist
                    newCenter = j
            if clusterInfo[i,0] != newCenter:
                clusterChanged = True
                clusterInfo[i,:] = newCenter,minDist#**2

        #step 2: update the centers
        for j in range(k):
            members = Data[np.where(clusterInfo[:,0]==j)[0]]  #filter the lines whose assigned cluster number is j
            centers[j] = np.mean(members,axis = 0)

    SSE = sum(clusterInfo[:,1])
    #print 'SSE %.4f,' %(SSE),
    for j in range(k):
        print 'cluster%d:%d,' %(j,clusterInfo[:,0].tolist().count(j)),

    return clusterInfo, SSE

def Random_Kmeans(inData, Label, k, rdNum = 10):
    smp_Num = len(inData)
    k = 2
    minSSE, accuracy = 1e100 , 0

    for i in range(rdNum):
        print '#%d kmeans:' %(i+1),
        clusterInfo, SSE = kmeans(inData,k)
        if SSE < minSSE:
            minSSE = SSE
            bestCluster = clusterInfo        

    purity = measurePurity(smp_Num, k, bestCluster, Label)
    print 'purity = ', purity

    return bestCluster, purity

def PCA_analysis(inData, Label, maxEigenNum):
    #pdb.set_trace()
    cov       = np.cov(inData.T)
    [U, S, V] = np.linalg.svd(cov)
    sortIdx   = (-S).argsort()
    var_sum   = np.sum(S)
    print 'the first 10 feature are: ', sortIdx[:10]
    print 'var_sum is ', var_sum,'\n\ntheir eigen-value are ', S[sortIdx[:10]]

    # Q2, how much dimensions are needed to retain at least 80% and 90% of th total variance respectively?  
    cumRtVar    = S[sortIdx[:]]*float(100)/var_sum
    cumRtVar    = np.cumsum(cumRtVar)
    print '\nthe cumulative retained accuracy is: ', cumRtVar[:10]
    
    # extract eigen-vectors and do classification
    if (maxEigenNum > len(S)):
        print 'maximum eigen-vector number couldnot larger than feature number, fix it to feature number. \n'
        maxEigenNum = len(S)

    pcaPurity  = []
    pcaTime    = []
    for i in range(maxEigenNum):
        t0 = float(time.clock())

        print '\n*** PCA: using feature ', sortIdx[0:(i+1)]
        Ureduce            = U[:,sortIdx[0:(i+1)]]
        DataReduce         = np.dot(inData, Ureduce)
        [clusterInfo, purity] = Random_Kmeans(DataReduce, Label, 2, 10)

        t1 = float(time.clock())
        print 'using time %.4f s.' % (t1-t0)

        pcaPurity.append(round(purity, 4))
        pcaTime.append(round(t1-t0, 4))
    return cumRtVar, pcaPurity, pcaTime

def get_colour(color):
    '''
    b---blue   c---cyan  g---green    k----black
    m---magenta r---red  w---white    y----yellow'''
    color_set = ['r','b','m','g','c','k','y']
    return color_set[color % 7]


def computeMeanCovariance(Data, Label, class_label):
    members = Data[np.where(Label == class_label)]
    mean = np.mean(members, 0)
    scatter = np.dot((members - mean).T, (members - mean))
    return mean, scatter

def LDA_analysis(Data, Label):
    [mean0, scatter0] = computeMeanCovariance(Data, Label, 0)
    [mean1, scatter1] = computeMeanCovariance(Data, Label, 1)
    S = scatter0 + scatter1
    w = np.dot(inv(S), (mean0 - mean1))

    return np.dot(Data, w)

def measurePurity(n_data, k, bestCluster, Label):
    labelPredict = np.zeros(n_data)
    for i in range(k):
        members = np.where(bestCluster[:,0] == i)[0]
        majorityClass = mode(Label[members])[0][0]
        labelPredict[members] = majorityClass
    error = sum(abs(labelPredict-Label))
    purity = 100 - 100.0*float(error)/n_data

    return purity

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

    purity = measurePurity(n_data, k, bestCluster, Label)
    t1 = float(time.clock())
    allF_purity = round(purity, 4)
    allF_time   = round(t1-t01, 4)
    print '[Kmeans] purity with best SSE is %.3f%%, time for part 1 is %.4f s. \n' % (purity,t1-t01)
    

    #******************Part 2***************************
    print '******Part 2******'
    t01 = float(time.clock())
    maxEigenNum = 3
    [curVar, pcaPurity, pcaTime] = PCA_analysis(Data, Label, maxEigenNum)

    print 'PCA purities are: ', pcaPurity, '\n runing times are: ', pcaTime

    #------------------- 
    plt.figure()
    t = np.arange(1, len(curVar)+1, 1)

    plt.subplot(211)
    plt.plot(t, curVar, 'b')

    plt.subplot(212)
    plt.plot(t[0:180], curVar[0:180], 'b')
    plt.xlabel('feature number')
    plt.ylabel('percentage of cumulative variance', position = (0.1, 1))
    plt.savefig(saveDir+'pca_cumVarPerc.png')
    plt.cla()
    #---------------------------
    plt.subplot()

    pcaPurity.append(allF_purity)
    pcaTime.append(allF_time)
    ind = np.arange(len(pcaPurity))
    width = 0.35
    
    rectsP = plt.bar(ind, pcaPurity, width, color = 'b', label = 'purity')
    rectsT = plt.bar(ind+width, pcaTime, width, color = 'r', label = 'time cost')
    plt.xlim(0, 5)
    plt.xlabel('feature selection')
    plt.xticks(ind+width, ('pca_1', 'pca_2', 'pca_3', 'all feature')) 
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(saveDir+'pca_comparing.png')

    #--------------------------------
    t1 = float(time.clock())
    print '[PCA] time for part 2 is %.4f s. \n' % (t1-t01)

    
    #******************Part 3***************************
    print '******Part 3******'
    t01 = float(time.clock())
    projected_data = LDA_analysis(Data, Label)
    print projected_data[0:6]
    bestCluster, purity = Random_Kmeans(projected_data, Label, 2, 10)
    #bestCluster, SSE = kmeans(projected_data,2)
    #purity = measurePurity(n_data, k, bestCluster, Label)
    t1 = float(time.clock())
    print '[LDA] purity with best SSE is %.3f%%, time for part 3 is %.4f s. \n' % (purity,t1-t01)


    t1 = float(time.clock())
    print '[done] total runtime is %.4f s. \n' % (t1-t00)
    

if __name__ == "__main__":
    RunMain()
