import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib # save classify model.
import time
import pdb
import random

import ExtractFeatures as ef
import ClassifyModels as cf

def FeatureExtraction():
	#[trainX, trainY, testX, testY] = ef.LoadData()

	trainFolder  = ""
	testFile  = ["./Data/diyDataset/test/1"]
	trainY = ["test"]
	print testFile[0] + ".wav"
	testF = ef.ExtractFeaturesByLibrosa(testFile, trainY, trainFolder)

	return testF

def Classify(testData, FileName, modelName):
	if modelName == 'GMM':
		gmmBag = []
		for i in range(2):
			clf = joblib.load(FileName+modelName+'_'+str(i)+'.pkl')
			gmmBag.append(clf)

			[gmmTest, gmmAcc] = cf.gmm_classify(testData, gmmBag)
		return gmmTest

	elif modelName == 'SVM':
		svmModel          = joblib.load(FileName+modelName+'.pkl')
		[svmTest, svmAcc] = cf.svm_classify(testData, svmModel)
		return svmTest

	elif modelName == 'NN':
		nnModel         = joblib.load(FileName+modelName+'.pkl')
		[nnTest, nnAcc] = cf.nn_classify(cfModel.norTestData, nnModel)
		return nnTest

	else: # modelName == 'pcp'
		pcpModel          = joblib.load(FileName+modelName+'.pkl')
		[pcpTest, pcpAcc] = cf.perceptron_classify(cfModel.norTestData, pcpModel)
		return pcpTest

def RunMain():
	# extractFeature from file to testFeature.
	#testData = 0  # even we don't know the label, hope could set it to a random value.
	testData = FeatureExtraction()
	testData = np.c_[testData, 0]

	FileName  = './Result/Model_'
	modelName = 'GMM'
	testRst = Classify(testData, FileName, modelName)

	className = ['Chinese', 'English']
	testRst_Name = className[testRst[0]]
	print testRst_Name


if __name__ == "__main__":
	RunMain()


