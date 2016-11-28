import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib # save classify model.
import time
import pdb
import random

import ExtractFeatures as ef
import ClassifyModels as cf

def Classify(testData, modelName):
	if modelName == 'GMM':
		gmmBag = []
		for i in range(classNum):
			clf = joblib.load(fileName +'gmmC'+str(i)+'.pkl')
			gmmBag.append(clf)

			[gmmTest, gmmAcc] = cf.gmm_classify(testData, gmmRdBag)
		return gmmTest

	elif modelName == 'SVM':
		svmModel          = joblib.load(fileName +'svm'+str(i)+'.pkl')
		[svmTest, svmAcc] = cf.svm_classify(testData, svmModel)
		return svmTest

	elif modelName == 'NN':
		nnModel         = joblib.load(fileName +'nn'+str(i)+'.pkl')
		[nnTest, nnAcc] = cf.nn_classify(cfModel.norTestData, nnModel)
		return nnTest

	else: # modelName == 'pcp'
		pcpModel          = joblib.load(fileName +'pcp'+str(i)+'.pkl')
		[pcpTest, pcpAcc] = cf.perceptron_classify(cfModel.norTestData, pcpModel)
		return pcpTest

def RunMain():
	# extractFeature from file to testFeature.
	testData = 0  # even we don't know the label, hope could set it to a random value.

	modelName = 'GMM'
	testRst = Classify(testData, modelName)


if __name__ == "__main__":
	RunMain()


	