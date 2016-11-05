#!/usr/bin/env python
import numpy as np
import sys
import operator
import pdb

'''
# for machine learning assignment3,
# read a matrix, 5 columns, with 1~4 as 4 features and 5th as label.
'''
def LoadData():
    # read input data for training & testing
    
    dataDir    = "./iris-data/"
    trainName  = "iris_train.csv"
    testName   = "iris_test.csv"
    trainData  = np.genfromtxt(dataDir+trainName, delimiter=";")
    testData   = np.genfromtxt(dataDir+testName,   delimiter=";")
    trainData = trainData[:,:]
    '''
    # extract data as Matrix / vector
    (dataNum, dataLen) = trainData.shape
    trainX = trainData[:,0:dataLen-1]
    trainY = trainData[:,dataLen-1]

    testX = testData[:,0:dataLen-1]
    testY = testData[:,dataLen-1]
    return [trainX, trainY, testX, testY]
    '''
    return trainData, testData

