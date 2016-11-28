import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib # save classify model.
import time
import pdb
import random

import ClassifyModels as cf


def FeatureAnalysisBasedData(cfModel):
	feaIdx  = [0, 12, 24, 36, 164, 184, 185, 186, 187, 194, 195, 197, 203, 204]
	feaName = ['stft', 'cqt', 'cens', 'mel-spec', 'mfcc', 'rmse', 'spec_centroid',\
	            'spec_bandwidth', 'spec_contrast', 'spec_rolloff', 'poly_features',\
	            'tonnetz', 'zero_crossing_rate']
	feaNum  = 13

	# test on each feature
	testAccList = []
	for i in range(feaNum):
		stIdx  = feaIdx[i]
		endIdx = feaIdx[i+1]

		# print i
		svmModel = cf.perceptron_train(cfModel.norTrainData, stIdx, endIdx)
		# print i
		[svmTest, svmAcc] = cf.perceptron_classify(cfModel.testData, svmModel, stIdx, endIdx)
		testAccList.append(round(svmAcc* float(100)/cfModel.testLen, 3))
	
	print '\n using feature group includes: \n', feaName
	print '\n number of feature in each grouop: \n', (np.array(feaIdx[1:])-np.array(feaIdx[:-1]))
	print '\n test on single feature group, accuracy is\n', testAccList

def FeatureReduction_PCA(cfModel, eigenThr):
	#def PCA_analysis(inData, Label, maxEigenNum):
	inData    = np.vstack(cfModel.trainData[:-1], cfModel.testData[:-1])
	#inData    = np.array(inData)
	cov       = np.cov(inData.T)
	[U, S, V] = np.linalg.svd(cov)
	sortIdx   = (-S).argsort()

	# Q2, how much dimensions are needed to retain at least 80% and 90% of th total variance respectively?
	cumRtVar    = S[sortIdx[:]]*float(100)/np.sum(S)
	cumRtVar    = np.cumsum(cumRtVar)
	eigenIdx    = [i for i in xrange(len(S)) if cumRtVar > eigenThr][0]

	# extract eigen-vectors and do classification
	print '\n*** PCA: using feature ', sortIdx[0:(eigenIdx+1)]
	Ureduce            = U[:,sortIdx[0:(eigenIdx+1)]]

	# classification on projection space
	trainData_rd  = np.dot(cfModel.trainData[:-1], Ureduce) # feature reduce data
	newTrainData  = np.hstack(DataReduce, cfModel.trainData[-1])

	testData_rd  = np.dot(cfModel.testData[:-1], Ureduce) # feature reduce data
	newTestData  = np.hstack(DataReduce, cfModel.testData[-1])

	# do training and classification.
	# .... 


def RunMain():
	time.clock()
	t0 = float(time.clock())

	DIR_RESULT = "./Result/"
	MODEL_NAME = "Model_"


	DIR        = "./Feature/"
	TRAIN_FILE = "train.dev"
	TEST_FILE  = "test.dev"

	cfModel = cf.ClassModel()
	cfModel.readFile(DIR+TRAIN_FILE, 1)
	cfModel.readFile(DIR+TEST_FILE, 0)

	cfModel.cropTrainData(50)
	cfModel.featureNormalize()

	classLabel = [0, 1] # 0-Chinese, 1-English
	#pdb.set_trace()

	# feature analysis.
	if 0:
		FeatureAnalysisBasedData(cfModel)

	# test different model over all feature.
	gmmBag = cf.gmm_train(cfModel.trainData, classLabel)
	[gmmTest, gmmAcc] = cf.gmm_classify(cfModel.testData, gmmBag)
	classNum = len(gmmBag)
	for i in range(classNum):
		joblib.dump(gmmBag[i], DIR_RESULT+MODEL_NAME+'gmmC' + str(i) +'.pkl')
	print "\nGMM test result:\n", "accuracy: ", round(gmmAcc* float(100)/cfModel.testLen,3) #, "\n test rst: ", gmmTest

	'''
	gmmRdBag = []
	for i in range(classNum):
		clf = joblib.load('gmmModel_c'+str(i)+'.pkl')
		gmmRdBag.append(clf)
	[gmmTest2, gmmAcc2] = cf.gmm_classify(cfModel.testData, gmmRdBag)
	print "\n2nd ** GMM test result:\n", "accuracy: ", round(gmmAcc* float(100)/cfModel.testLen,3) #, "\n test rst: ", gmmTest
	'''

	svmModel = cf.svm_train(cfModel.trainData)
	[svmTest, svmAcc] = cf.svm_classify(cfModel.testData, svmModel)
	joblib.dump(svmModel, DIR_RESULT+MODEL_NAME+'svm' +'.pkl')
	print "\nsvm test result:\n", "accuracy: ", round(svmAcc* float(100)/cfModel.testLen,3) #, "\n test rst: ", svmTest
	
	nnModel = cf.nn_train(cfModel.norTrainData)
	[nnTest, nnAcc] = cf.nn_classify(cfModel.norTestData, nnModel)
	joblib.dump(nnModel, DIR_RESULT+MODEL_NAME+'nn' +'.pkl')
	print "\nNN test result:\n", "accuracy: ", round(nnAcc* float(100)/cfModel.testLen,3)#, "\n test rst: ", nnTest
	
	pcpModel = cf.perceptron_train(cfModel.norTrainData)
	[pcpTest, pcpAcc] = cf.perceptron_classify(cfModel.norTestData, pcpModel)
	joblib.dump(pcpModel, DIR_RESULT+MODEL_NAME+'pcp' +'.pkl')
	print "\nPerceptron test result:\n", "accuracy: ", round(pcpAcc* float(100)/cfModel.testLen,3)#, "\n test rst: ", pcpTest
	
	# save trained model or models.



if __name__ == "__main__":
	RunMain()


