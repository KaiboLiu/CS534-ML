#!/usr/bin/env python
import numpy as np
import sys
import operator
#import pdb

'''
# for machine learning assignment2,
# read a string file, and convert string to integer.
# there would be different number of words in each line
'''
def LoadData_vocabulary(filename):
    vocList = [x.strip() for x in open(filename, "r").readlines()]
    wordNum = len(vocList)

    del(vocList)
    return wordNum

def LoadData_bagOfWords(filename):
    # string list to integer list
    docList_int = []
    docList = [x.strip() for x in open(filename, "r").readlines()]
    docNum  = len(docList)
    for doc in docList:
        words = doc.split()
        wordList_int = []
        for w in words:
            intW = int(w)
            wordList_int.append(intW)

        wordList_int.sort()
        docList_int.append(wordList_int)

    del(docList)
    return docList_int, docNum


'''
# for machine learning assignment2,
# read .dev file, and convert to 0/1 vector.
# there would a class name in each line
'''
def LoadData_labels(filename, str0):
    docList = [x.strip() for x in open(filename, "r").readlines()]

    labelY   = []
    for word in docList:
        if(word == str0):
            classY = 0
        else:
            classY = 1
        labelY.append(classY)

    del(docList)
    return labelY

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

