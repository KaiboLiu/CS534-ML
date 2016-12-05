import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib # save classify model
from matplotlib.colors import ListedColormap #color map.
from sklearn.ensemble import  AdaBoostClassifier
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


	#RandomForest $ AdaBoost
	n_estimators=100
	learning_rate=1.
	clf = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate,algorithm='SAMME.R')
	x_train = np.copy(trainData[:,:-1])
	y_train = np.copy(trainData[:,-1])
	clf.fit(x_train,y_train)
	result = clf.predict(testData[:,:-1])
	accuracy = 0
	n_test = len(testData)
	adaTestMat = np.array([[0,0],[0,0]])
	for i in range(n_test):
	    if result[i] == testData[i,-1]:
	        accuracy += 1
		adaTestMat[result[i], testData[i,-1]] = adaTestMat[result[i], testData[i,-1]]+1
	adaAcc = round(100*float(accuracy)/n_test,3)
	print "\n Adaboost Accuracy:", adaAcc, "\n confidence matrix:", adaTestMat
    #print float(match)/n_test,clf.score(testData[:,:-1],testData[:,-1])

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
	accuracy = [gmmAcc, svmAcc, nnAcc, pcpAcc, adaAcc]
	testMat  = np.c_[gmmTestMat, svmTestMat, nnTestMat, pcpTestMat, adaTestMat]
	return  accuracy, testMat

def BasicModelCompare(cfModel, modelName, saveDir):
	trainPerp = np.linspace(0.1, 1, 10)

	trainLen = len(cfModel.trainData)
	testRcd_acc = []
	for p in trainPerp:
		dsmpTrainData  = cfModel.trainData[0:np.int(p*trainLen)]
		[acc, testMat] = BasicModelAnalysis(dsmpTrainData, cfModel.testData)
		testRcd_acc.append(acc)

	#------------------------
	plt.figure()
	testRcd_acc = np.array(testRcd_acc)
	modelNum    = len(modelName)
	for i in range(modelNum):
		plt.plot(trainPerp, testRcd_acc[:,i], label=modelName[i])

	plt.legend(loc="center right")
	plt.xlabel("Proportion train")
	plt.ylabel("Test Accuracy(\%)")
	plt.savefig(saveDir + 'basicModelCompare.png')
	#-----------------------------

def BasicModelCompare_Draw(trainData, testData, modelName, saveDir):

	if(trainData.shape[1]!= 3):
		print 'feature number of each example should be 2. \n'
		return
	elif(trainData.shape[0] < 10):
		print 'need more data in training set. \n'
		return

	h = .02
	trX_min, trX_max = trainData[:, 0].min() - 0.05, trainData[:, 0].max() + 0.05
	trY_min, trY_max = trainData[:, 1].min() - 0.05, trainData[:, 1].max() + 0.05
	teX_min, teX_max = testData[:, 0].min() - 0.05, testData[:, 0].max() + 0.05
	teY_min, teY_max = testData[:, 1].min() - 0.05, testData[:, 1].max() + 0.05
	xx, yy = np.meshgrid(np.arange(min(trX_min, teX_min), max(trX_max, teX_max), h),
	                     np.arange(min(trY_min, teY_min), max(trY_max, teY_max), h))
	allData = np.c_[xx.ravel(), yy.ravel()]

	cm = plt.cm.RdBu
	cm_bright  = ListedColormap(['#FF0000', '#00FF00'])
	markerSign = ['+', 'o']
	trainSign  = [markerSign[np.int(e)] for e in trainData[:,2]]
	testSign   = [markerSign[np.int(e)] for e in testData[:,2]]

	plt.figure()
	modelNum = len(modelName)

	for i in range(3):
		plt.subplot(3, 1, i+1)
		# Plot the training points
		plt.scatter(trainData[:, 0], trainData[:, 1], c=trainData[:,2], cmap=cm_bright)
		# and testing points
		plt.scatter(testData[:, 0], testData[:, 1], c=testData[:,2], cmap=cm_bright, alpha=1)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.xticks(())
		plt.yticks(())

		#plt.title(modelName[i])

		# draw dicision boundary
		if i == 0: # svm
			model = cf.svm_train(trainData)
			Z = model.predict_proba(allData)[:, 1]
		elif i == 1: # nn
			model = cf.nn_train(trainData)
			Z = model.predict_proba(allData)[:, 1]
		elif i == 2: # perceptron
			model = cf.perceptron_train(trainData)
			Z = model.predict(allData)
		else: # gmm
			[Z, acc, mat] = cf.TrainAndClassify(trainData, np.c_[xx.ravel(), yy.ravel()], 'GMM')

		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, cmap=cm, alpha=.4)
	plt.savefig(saveDir+'basicModelDraw'+str(i)+'.png')

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
    #--------------------------------

def FeatureReduction_PCA(cfModel, eigenThr, modelName, maxEigenNum):
	inData    = np.vstack((cfModel.trainData[:,:-1], cfModel.testData[:,:-1]))
	#inData    = np.array(inData)
	cov       = np.cov(inData.T)
	[U, S, V] = np.linalg.svd(cov)
	sortIdx   = (-S).argsort()

	# Q2, how much dimensions are needed to retain at least 80% and 90% of th total variance respectively?
	cumRtVar    = S[sortIdx[:]]*float(100)/np.sum(S)
	cumRtVar    = np.cumsum(cumRtVar)
	eigenIdx    = maxEigenNum #200 # [i for i in xrange(len(S)) if cumRtVar[i] > eigenThr][0]
	eigenIdx = eigenIdx.astype(int)
	feaName     = fr.match_with_features(sortIdx[0:(eigenIdx+1)])
	#print 'eigen feature name: ', feaName

	# extract eigen-vectors and do classification
	#print '\n*** PCA: using feature ', sortIdx[0:(eigenIdx+1)]
	Ureduce            = U[:,sortIdx[0:(eigenIdx+1)]]

	# classification on projection space
	trainData_rd  = np.dot(cfModel.trainData[:,:-1], Ureduce) # feature reduce data
	newTrainData  = np.c_[trainData_rd, cfModel.trainData[:,-1]]

	testData_rd  = np.dot(cfModel.testData[:,:-1], Ureduce) # feature reduce data
	newTestData  = np.c_[testData_rd, cfModel.testData[:,-1]]

	# do training and classification.
	# if use NN, Perception, it's better to do feature normalize first.
	[newTrainData, newTestData] = cf.featureNormalize(newTrainData, newTestData)
	[testRst, accuracy, model, testMat] = cf.TrainAndClassify(newTrainData, newTestData, 'NN')
	#print "\nPCA & GMM test result:\n", "accuracy: ", accuracy

	return testRst, accuracy

def FeatureReduction_sklearn(cfModel, modelName):
	trainX = cfModel.trainData[:,:-1]
	trainY = cfModel.trainData[:,-1]
	testX  = cfModel.testData[:,:-1]
	testY  = cfModel.testData[:,-1]

	# feature selection with LDA
	[new_trainX, new_testX, feaNum, featureName]= fr.feature_selection_lda(trainX, trainY, testX)
	new_testX = new_testX[:, np.newaxis]
	new_trainData = np.c_[new_trainX.T, trainY]
	new_testData  = np.c_[new_testX, testY]

	[testRst, accuracy, model, testMat] = cf.TrainAndClassify(new_trainData, new_testData, modelName)
	#print "\nsklearn test result:\n", "accuracy: ", accuracy
	plt.figure()
	plt.plot(1, accuracy, 'ro', label='LDA')

	# feature selection with high variance
	accuraceList = []
	X = []
	R = np.linspace(0, 1, 20)
	for i in R:
		[new_trainX, new_testX, feaNum, featureName]= fr.feature_selection_var(trainX, trainY, testX, i)
		new_trainData = np.c_[new_trainX, trainY]
		new_testData  = np.c_[new_testX, testY]
		[testRst, accuracy, model, testMat] = cf.TrainAndClassify(new_trainData, new_testData, modelName)
		accuraceList.append(accuracy)
		X.append(feaNum)
		#print "\nsklearn test result:\n", "accuracy: ", accuracy
	X = np.array(X)
	sIdx = X.argsort()
	X = X[sIdx]
	accuraceList = np.array(accuraceList)
	accuraceList.astype(int)
	accuraceList = accuraceList[sIdx.tolist()]

	d = {}
	for a, b in zip(X, accuraceList):
		d.setdefault(a, []).append(b)
	X = []
	Y = []
	for key in d:
		X.append(key)
		Y.append(sum(d[key])/len(d[key]))

	plt.plot(X, Y, 'r', label='variance')
	plt.xlabel('number of features')
	plt.ylabel('accuracy')

	# feature selection with L1 norm
	accuraceList = []
	X = []
	R = np.linspace(0.0001, 10000, 20)
	for i in R:
		[new_trainX, new_testX, feaNum, featureName]= fr.feature_selection_L1(trainX, trainY, testX, i)
		new_trainData = np.c_[new_trainX, trainY]
		new_testData  = np.c_[new_testX, testY]
		[testRst, accuracy, model, testMat] = cf.TrainAndClassify(new_trainData, new_testData, modelName)
		accuraceList.append(accuracy)
		X.append(feaNum)
		#print "\nsklearn test result:\n", "accuracy: ", accuracy
	X = np.array(X)
	sIdx = X.argsort()
	X = X[sIdx]
	accuraceList = np.array(accuraceList)
	accuraceList.astype(int)
	accuraceList = accuraceList[sIdx.tolist()]
	plt.plot(X.tolist(), accuraceList.tolist(),'b', label='L1')
	plt.xlabel('number of features')
	plt.ylabel('accuracy')

	# feature selection with random forest
	accuraceList = []
	X = []
	R = np.linspace(1, 204, 20)
	for i in R:
		[new_trainX, new_testX, feaNum, featureName]= fr.feature_selection_tree(trainX, trainY, testX, i)
		new_trainData = np.c_[new_trainX, trainY]
		new_testData  = np.c_[new_testX, testY]
		[testRst, accuracy, model, testMat] = cf.TrainAndClassify(new_trainData, new_testData, modelName)
		accuraceList.append(accuracy)
		X.append(feaNum)
		#print "\nsklearn test result:\n", "accuracy: ", accuracy
	X = np.array(X)
	sIdx = X.argsort()
	X = X[sIdx]
	accuraceList = np.array(accuraceList)
	accuraceList.astype(int)
	accuraceList = accuraceList[sIdx.tolist()]
	plt.plot(X.tolist(), accuraceList.tolist(),'g', label='Tree')
	plt.xlabel('number of features')
	plt.ylabel('accuracy')

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
	modelName   = ['linear SVM', 'Neural Network', 'linear perceptron', 'GMM']
	trainData = np.c_[cfModel.trainData[:,0:2], cfModel.trainData[:,-1]]
	testData  = np.c_[cfModel.testData[:,0:2], cfModel.testData[:,-1]]
	BasicModelCompare_Draw(trainData, testData, modelName, DIR_RESULT)

	modelName   = ['GMM', 'linear SVM', 'NN', 'linear Perceptron', 'AdaBoost']
	BasicModelCompare(cfModel, modelName, DIR_RESULT)
	BasicModelAnalysis(cfModel.trainData, cfModel.testData, DIR_RESULT)

	# feature analysis.
	FeatureAnalysisBasedData(cfModel, DIR_RESULT)

	[testRst, acc] = FeatureReduction_sklearn(cfModel, 'GMM')

	accuraceList = []
	X = []
	R = np.linspace(1, 204, 20)
	for i in R:
		[testRst, acc] = FeatureReduction_PCA(cfModel, 90, 'GMM', i)
		accuraceList.append(acc)
		X.append(i)
	print X
	print accuraceList
	plt.plot(X, accuraceList,'m', label='PCA')
	plt.xlabel('number of features')
	plt.ylabel('accuracy')
	plt.legend(loc='lower left')
	plt.tight_layout()
	plt.savefig(DIR_RESULT+'featureReu.png')

	plt.show()
	plt.close('all')

if __name__ == "__main__":
	RunMain()
