#!/usr/bin/env python
import numpy as np

'''
# for machine learning assignment4,
# read a matrix, 2059*477 as Data, and a list 2059*1 as Label.
'''
def LoadData():
    # read input data for training & testing
    
    dataDir    = "./walking-data/"
    dataName  = "walking.train.data"
    labelName   = "walking.train.labels"
    '''
    #genfromtxt is slower than open+readlines
    Data  = np.genfromtxt(dataDir+dataName, delimiter=" ")
    Label   = np.genfromtxt(dataDir+labelName, delimiter=" ")
    '''
    Data , Label = [],[]
    fileIn = open(dataDir+dataName)
    for line in fileIn.readlines():  
        Data.append(map(float,line.strip().split(' ')))   #convert str to float
    Data = np.array(Data)

    fileIn = open(dataDir+labelName)
    for line in fileIn.readlines(): 
        #Label.append(map(int,line.strip().split(' ')))    #convert str to int
        Label.append(int(line.strip()))    #convert str to int
    Label = np.array(Label)
    print 'data size:',Data.shape
    return Data, Label

