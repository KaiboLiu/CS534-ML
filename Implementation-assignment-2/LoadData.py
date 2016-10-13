#!/usr/bin/env python
import numpy as np

'''
# for machine learning assignment2,
# read a string file, and convert string to integer.
# there would be different number of words in each line
'''
def LoadData_bagOfWords():

'''
# for machine learning assignment2,
# read .dev file, and convert to 0/1 vector.
# there would a class name in each line
'''
def LoadData_dev():

'''
 # for machine learning assignment1,
 # read .csv file. X is arranged as a matrix, Y is a vector.
'''
def LoadData_csv():
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

