#!/usr/bin/env python
import numpy as np
import time

def LoadData():
    # read input data for training & testing
    dataDir    = "./"
    trainName  = "train p1-16.csv"
    testName   = "test p1-16.csv"
    trainData  = np.genfromtxt(dataDir+trainName, delimiter=",")
    testData   = np.genfromtxt(dataDir+testName,   delimiter=",")

    # extract data as Matrix / vector
    (dataNum, dataLen) = trainData.shape
    trainX = trainData[:,0:dataLen-1]
    trainY = trainData[:,dataLen-1]

    testX = testData[:,0:dataLen-1]
    testY = testData[:,dataLen-1]
    return [trainX, trainY, testX, testY]

