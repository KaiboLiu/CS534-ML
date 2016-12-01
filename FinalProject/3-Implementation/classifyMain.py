import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib # save classify model.
import time
import pdb
import random

import ClassifyModels as cf
import FeatureReduction as fr

def BasicModelAnalysis(trainData, testData, saveDir = './', doSave = 0):

	testLen = len(testData)
	[norTrainData, norTestData] = cf.featureNormalize(trainData, testData)

	# test different model over all feature.
	[gmmTest, gmmAcc, gmmBag, gmmTestMat] = cf.TrainAndClassify(norTrainData, norTestData, 'GMM')
	print "\nGMM test result:\n", "accuracy: ", gmmAcc #, "\n test rst: ", gmmTest
	print 'test number:', testLen, 'testMat is: \n', gmmTestMat
	
	[svmTest, svmAcc, svmModel, svmTestMat] = cf.TrainAndClassify(norTrainData, norTestData, 'SVM')
	print "\nsvm test result:\n", "accuracy: ", svmAcc #, "\n test rst: ", svmTest
	print svmTestMat
	
	# pdb.set_trace()
	[nnTest, nnAcc, nnModel, nnTestMat] = cf.TrainAndClassify(norTrainData, norTestData, 'NN')
	print "\nNN test result:\n", "accuracy: ", nnAcc#, "\n test rst: ", nnTest
	print nnTestMat
	

	[pcpTest, pcpAcc, pcpModel, pcpTestMat] = cf.TrainAndClassify(norTrainData, norTestData, 'Perceptron')
	print "\nPerceptron test result:\n", "accuracy: ", pcpAcc#, "\n test rst: ", pcpTest
	print pcpTestMat

	if doSave == 1:
		cf.saveModel(gmmBag,   'GMM', saveDir, 'Model_')
		cf.saveModel(svmModel, 'SVM', saveDir, 'Model_')
		cf.saveModel(nnModel,  'NN', saveDir, 'Model_')
		cf.saveModel(pcpModel, 'Perceptron', saveDir, 'Model_')

	'''
	# test about model read.
	gmmRdBag = []
	for i in range(2):
		clf = joblib.load(DIR_RESULT+'Model_'+'GMM_'+str(i)+'.pkl')
		gmmRdBag.append(clf)
	pdb.set_trace()
	[gmmTest2, gmmAcc2, testMat] = cf.gmm_classify(testData, gmmRdBag)
	print "\n2nd ** GMM test result:\n", "accuracy: ", gmmAcc #, "\n test rst: ", gmmTest
	print 'test number:', testLen, 'testMat is: \n', testMat
	'''

	accuracy = [gmmAcc, svmAcc, nnAcc, pcpAcc]
	testMat  = np.c_[gmmTestMat, svmTestMat, nnTestMat, pcpTestMat]
	return  accuracy, testMat

def BasicModelCompare(cfModel, saveDir):
	trainPerp = np.linspace(0.1, 1, 10)

	trainLen = len(cfModel.trainData)
	testRcd_acc = []
	for p in trainPerp:
		dsmpTrainData  = cfModel.trainData[0:np.int(p*trainLen)]
		[acc, testMat] = BasicModelAnalysis(dsmpTrainData, cfModel.testData)
		testRcd_acc.append(acc)

	#------------------------
	plt.figure()
	modelName   = ['GMM', 'NN', 'linear SVM', 'linear Perceptron']
	testRcd_acc = np.array(testRcd_acc)
	modelNum    = testRcd_acc.shape[1]
	for i in range(modelNum):
		plt.plot(trainPerp, testRcd_acc[:,i], label=modelName[i])

	plt.legend(loc="center right")
	plt.xlabel("Proportion train")
	plt.ylabel("Test Accuracy(\%)")
	plt.savefig(saveDir + 'basicModelCompare.png')
	# plt.show()
	#-----------------------------


def FeatureAnalysisBasedData(cfModel, saveDir):
	feaIdx  = [0, 12, 24, 36, 164, 184, 185, 186, 187, 194, 195, 197, 203, 204]
	feaName = ['stft', 'cqt', 'cens', 'mel-spec', 'mfcc', 'rmse', 'spec_centroid',\
	            'spec_bw', 'spec_contrast', 'spec_rolloff', 'poly_features',\
	            'tonnetz', 'zero_cross_rate']
	feaNum  = 13

	# test on each feature
	testAccList = []
	for i in range(feaNum):
		stIdx  = feaIdx[i]
		endIdx = feaIdx[i+1]
		newTrainData = np.c_[cfModel.norTrainData[:,stIdx:endIdx], cfModel.norTrainData[:,-1]]
		newTestData  = np.c_[cfModel.norTestData[:,stIdx:endIdx], cfModel.norTestData[:,-1]]

		# pdb.set_trace()
		[testRst, accuracy, model, testMat] = cf.TrainAndClassify(newTrainData, newTestData, 'SVM')
		testAccList.append(accuracy)

	feaLen = (np.array(feaIdx[1:])-np.array(feaIdx[:-1]))
	print '\n using feature group includes: \n', feaName
	print '\n number of feature in each grouop: \n', feaLen
	print '\n test on single feature group, accuracy is\n', testAccList

	#---------------------------
	plt.figure()
	ind = np.arange(len(testAccList))
	width = 0.40

	rectsP = plt.bar(ind, feaLen, width, color = 'b', label = 'number of features')
	rectsT = plt.bar(ind+width, testAccList, width, color = 'r', label = 'test accuracy')
	plt.xlabel('feature group')
	plt.xticks(ind, feaName, rotation=45)
	plt.legend()

	plt.tight_layout()
	plt.savefig(saveDir+'eatureAna_seprate.png')
	#plt.show()
    #--------------------------------
    

def FeatureReduction_PCA(cfModel, eigenThr, modelName):
	inData    = np.vstack((cfModel.trainData[:,:-1], cfModel.testData[:,:-1]))
	#inData    = np.array(inData)
	cov       = np.cov(inData.T)
	[U, S, V] = np.linalg.svd(cov)
	sortIdx   = (-S).argsort()

	# Q2, how much dimensions are needed to retain at least 80% and 90% of th total variance respectively?
	cumRtVar    = S[sortIdx[:]]*float(100)/np.sum(S)
	cumRtVar    = np.cumsum(cumRtVar)
	eigenIdx    = [i for i in xrange(len(S)) if cumRtVar[i] > eigenThr][0]

	# extract eigen-vectors and do classification
	print '\n*** PCA: using feature ', sortIdx[0:(eigenIdx+1)]
	Ureduce            = U[:,sortIdx[0:(eigenIdx+1)]]

	# pdb.set_trace()
	# classification on projection space
	trainData_rd  = np.dot(cfModel.trainData[:,:-1], Ureduce) # feature reduce data
	newTrainData  = np.c_[trainData_rd, cfModel.trainData[:,-1]]

	testData_rd  = np.dot(cfModel.testData[:,:-1], Ureduce) # feature reduce data
	newTestData  = np.c_[testData_rd, cfModel.testData[:,-1]]

	# do training and classification.
	# if use NN, Perception, it's better to do feature normalize first.
	[testRst, accuracy, model, testMat] = cf.TrainAndClassify(newTrainData, newTestData, 'NN')
	print "\nPCA & GMM test result:\n", "accuracy: ", accuracy

	return testRst, accuracy

def FeatureReduction_sklearn(cfModel):
	trainX = cfModel.trainData[:,:-1]
	trainY = cfModel.trainData[:,-1]
	testX  = cfModel.testData[:,:-1]
	testY  = cfModel.testData[:,-1]


	[new_trainX, new_testX, feaNum, featureName]= fr.feature_selection_lda(trainX, trainY, testX, testY)
	new_trainData = np.c_[new_trainX, trainY]
	new_testData  = np.c_[new_testX, testY]

	[testRst, accuracy, model, testMat] = cf.TrainAndClassify(new_trainData, new_testData, 'GMM')
	print "\nsklearn test result:\n", "accuracy: ", accuracy

	'''
	accuraceList = []

	L = np.arange(0.5,1.01,0.1)
	for i in range(len(L)):
		print L[i]
		[new_trainX, new_testX, feaNum, featureName]= fr.feature_selection_var(trainX, trainY, testX, testY, L[i])
		new_trainData = np.c_[new_trainX, trainY]
		new_testData  = np.c_[new_testX, testY]

		[testRst, accuracy, model, testMat] = cf.TrainAndClassify(new_trainData, new_testData, 'GMM')
		accuraceList.append(round(accuracy* float(100)/cfModel.testLen,3))
		print "\nsklearn test result:\n", "accuracy: ", accuracy
	plt.plot(accuraceList)
	plt.show()

	L = np.arange(0.01,100,10)
	for i in range(len(L)):
		print L[i]
		[new_trainX, new_testX, feaNum, featureName]= fr.feature_selection_L1(trainX, trainY, testX, testY, L[i])
		new_trainData = np.c_[new_trainX, trainY]
		new_testData  = np.c_[new_testX, testY]

		[testRst, accuracy, model, testMat] = cf.TrainAndClassify(new_trainData, new_testData, 'GMM')
		accuraceList.append(round(accuracy* float(100)/cfModel.testLen,3))
		print "\nsklearn test result:\n", "accuracy: ", accuracy

	plt.plot(accuraceList)
	'''
	plt.show()

	return testRst, accuracy

def RunMain():
	time.clock()
	t0 = float(time.clock())

	DIR_RESULT = "./Result/"
	DIR        = "./Feature/"
	TRAIN_FILE = "train.dev"
	TEST_FILE  = "test.dev"

	cfModel = cf.ClassModel()
	cfModel.readFile(DIR+TRAIN_FILE, 1)
	cfModel.readFile(DIR+TEST_FILE, 0)

	# cfModel.cropTrainData(100)
	cfModel.featureNormalize()

	classLabel = [0, 1] # 0-Chinese, 1-English

	# basic model comparison
	BasicModelCompare(cfModel, DIR_RESULT)
	BasicModelAnalysis(cfModel.trainData, cfModel.testData, DIR_RESULT)

	# feature analysis.
	FeatureAnalysisBasedData(cfModel, DIR_RESULT)

	[testRst, acc] = FeatureReduction_PCA(cfModel, 90, 'GMM')
	[testRst, acc] = FeatureReduction_sklearn(cfModel)



if __name__ == "__main__":
	RunMain()


