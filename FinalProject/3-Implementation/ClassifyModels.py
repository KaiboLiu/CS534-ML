##############################################
'''
CS534--Machine Learning Final Project.  Nov. 6th, 2016
Classify Part: input: traning examples with features and label
                      testing examples with features and label
               output: test result.
Models: GMM / 
'''
###############################################
import numpy as np
import itertools

from scipy import linalg
from sklearn import mixture # GMM model
from sklearn import svm     # svm model
from sklearn.neural_network import MLPClassifier # NN model
from sklearn.linear_model import perceptron  # perceptron
from sklearn.preprocessing import StandardScaler # for normalize
from sklearn.externals import joblib # save classify model.
import collections
import random
import copy
import matplotlib.pyplot as plt
import pdb
# import matplotlib.pyplot as plt
# import matplotlib as mpl


class ClassModel:
	def __init__(self, trainData=[], testData=[]):
		self.dsmpTrainData = trainData
		self.trainData     = trainData
		self.testData      = testData
		self.norTrainData  = []
		self.norTestData   = []

		self.testLen = len(self.testData)

	def readFile(self, fileName, isTrain):
		if isTrain == True:
			self.trainData     = np.genfromtxt(fileName)
			self.dsmpTrainData = self.trainData
		else:
			self.testData    = np.genfromtxt(fileName)
			self.testLen = len(self.testData)

	def cropTrainData(self, number):
		smp_num = len(self.trainData)
		if number > smp_num:
			number = smp_num
		rdm_idx = random.sample(xrange(0,smp_num), smp_num)
		self.trainData = np.delete(self.trainData, range(number,smp_num), 0)
		self.testData = np.vstack((self.testData,self.trainData[0:number]))
		self.testLen = len(self.testData)
		

	def divideTrainData(self, percent):
		train_size = len(self.trainData)
		sset_size  = np.int(train_size*percent)
		sset_idx   = random.sample(range(train_size), sset_size)
		train_sset = self.trainData[sset_idx[:]]

		val_idx  = set(range(train_size)) - set(sset_idx)
		val_sset = self.trainData[val_idx[:]]

		return train_sset, val_sset

	def featureNormalize(self):
		scaler = StandardScaler()
		scaler.fit(self.trainData[:,0:-1])

		self.norTrainData         = self.trainData
		self.norTrainData[:,0:-1] = scaler.transform(self.trainData[:,0:-1])

		self.norTestData          = self.testData
		self.norTestData[:,0:-1]  = scaler.transform(self.testData[:,0:-1])
		'''
		del self.norTrainData[:]
		del self.norTestData[:]

		smp_num, fea_num = np.shape(trainData[:,0:-1])
		for k in range(fea_num):
			Xmean = np.mean(self.trainData[:,k])
			Xstd  = np.std(self.trainData[:,k])
			if Xstd == 0: # protection for divide by 0
				Xstd = 1
			self.norTrainData[:,k] = (self.trainData[:,k]-Xmean)/Xstd
			self.norTestData[:,k]  = (self.testData[:,k] -Xmean)/Xstd
		'''
def featureNormalize(trainData, testData):
	scaler = StandardScaler()
	scaler.fit(trainData[:,0:-1])

	norTrainData         = trainData
	norTrainData[:,0:-1] = scaler.transform(trainData[:,0:-1])

	norTestData          = testData
	norTestData[:,0:-1]  = scaler.transform(testData[:,0:-1])
	return norTrainData, norTestData

def gmm_train(trainData, classLabel, feaSt = 0, feaEnd = -1, maxModelNum = 5):
	# statistic class number and examples.
	class_trainData = []
	classNum        = 0
	for k in classLabel:
		tmp = [ele[feaSt: feaEnd] for ele in trainData if ele[-1] == k]
		class_trainData.append(tmp)
		classNum = classNum + 1

	# GMM fit
	bestGMM  = []
	cv_types = ['spherical']#, 'tied', 'diag', 'full']
	for k in range(classNum):
		lowest_bic = np.infty
		train_bic = [] # cost, to check fitness
		for cv_type in cv_types:
			for n_components in range(1, maxModelNum):
				# Fit a Gaussian mixture with EM
				gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
				gmm.fit(class_trainData[k])

				train_bic.append(gmm.bic(np.array(class_trainData[k])))
				if train_bic[-1] < lowest_bic:
					lowest_bic = train_bic[-1]
					best_gmm   = gmm
		bestGMM.append(best_gmm)

	return bestGMM

def gmm_classify(testData, modelBag, feaSt = 0, feaEnd = -1):
	classNum = len(modelBag)
	
	# test on GMM model
	testResult = []
	testMat    = np.array([[0,0],[0,0]])
	accuracy = 0
	for row in testData:
		bestScore = -np.infty
		bestLabel = None
		for k in range(classNum):
			score = modelBag[k].score(row[feaSt: feaEnd].reshape(1,-1)) # how score looks like?
			if(score > bestScore):
				bestScore = score
				bestLabel = k
		if bestLabel == row[-1]:
			accuracy += 1
		testResult.append(bestLabel)
		testMat[bestLabel, row[-1]] = testMat[bestLabel, row[-1]]+1

	accuracy = round(accuracy * float(100)/len(testData),3)
	return testResult, accuracy, testMat

def svm_train(trainData, feaSt = 0, feaEnd = -1):
	# the changing param could be kernel, gamma, C.

	bestSVM = []
	# training data to fit model
	kernel = ['linear'] #, 'rbf','poly']
	bestScore = -np.infty
	for k in kernel:
		clf = svm.SVC(kernel = k, probability=True)
		clf.fit(trainData[:,feaSt: feaEnd], trainData[:,-1])

		score = clf.score(trainData[:, feaSt: feaEnd], trainData[:,-1])
		if(score > bestScore):
			bestScore = score
			bestSVM   = clf
	return bestSVM

def svm_classify(testData, svmModel, feaSt = 0, feaEnd = -1):
	# test on testData
	#pdb.set_trace()
	accuracy = 0
	testMat    = np.array([[0,0],[0,0]])
	testResult = []
	for row in testData:
		testRst = svmModel.predict(row[feaSt: feaEnd].reshape(1,-1))[0]
		if testRst == row[-1]:
			accuracy += 1
		testResult.append(testRst)
		testMat[testRst, row[-1]] = testMat[testRst, row[-1]]+1

	accuracy = round(accuracy * float(100)/len(testData),3)
	return testResult, accuracy, testMat

def nn_train(trainData, feaSt = 0, feaEnd = -1):
	# the varies parameter could be hidden layer info, 
	clf = MLPClassifier(activation='relu',solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (20, 2), random_state = 1)
	clf.fit(trainData[:,feaSt: feaEnd], trainData[:,-1])
	'''
	fig, axes = plt.subplots(4, 1)
	# use global min / max to ensure all weights are shown on the same scale
	vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
	for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
		ax.matshow(coef.reshape(4, 51), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
		ax.set_xticks(())
		ax.set_yticks(())

	plt.show()
	'''

	return clf

def nn_classify(testData, nnModel, feaSt = 0, feaEnd = -1):
	# test on testData
	accuracy   = 0
	testMat    = np.array([[0,0],[0,0]])
	testResult = []
	for row in testData:
		testRst = nnModel.predict(row[feaSt: feaEnd].reshape(1,-1))[0]
		if testRst == row[-1]:
			accuracy += 1
		testResult.append(testRst)
		testMat[testRst, row[-1]] = testMat[testRst, row[-1]]+1

	accuracy = round(accuracy * float(100)/len(testData),3)
	return testResult, accuracy, testMat

def perceptron_train(trainData, feaSt = 0, feaEnd = -1):
	clf = perceptron.Perceptron(n_iter = 15, shuffle = False, verbose = 0, random_state = None, fit_intercept = True)
	clf.fit(trainData[:, feaSt: feaEnd], trainData[:,-1])

	return clf

def perceptron_classify(testData, pcpModel, feaSt = 0, feaEnd = -1):
	# test on testData
	accuracy = 0
	testResult = []
	testMat    = np.array([[0,0],[0,0]])
	for row in testData:
		testRst = pcpModel.predict(row[feaSt: feaEnd].reshape(1,-1))[0]
		if testRst == row[-1]:
			accuracy += 1
		testResult.append(testRst)
		testMat[testRst, row[-1]] = testMat[testRst, row[-1]]+1

	accuracy = round(accuracy * float(100)/len(testData),3)
	return testResult, accuracy, testMat

def TrainAndClassify(trainData, testData, modelName, classLabel = [0, 1] ):
	if modelName == 'GMM':
		lnModel = gmm_train(trainData, classLabel)
		[testRst, acc, testMat] = gmm_classify(testData, lnModel)
	elif modelName == 'SVM':
		lnModel = svm_train(trainData)
		[testRst, acc, testMat] = svm_classify(testData, lnModel)
	elif modelName == 'NN':
		lnModel = nn_train(trainData)
		[testRst, acc, testMat] = nn_classify(testData, lnModel)
	else: # Perceptron
		lnModel = perceptron_train(trainData)
		[testRst, acc, testMat] = perceptron_classify(testData, lnModel)

	return testRst, acc, lnModel, testMat

def saveModel(model, modelName, dirName, fileName):
	if modelName == 'GMM':
		classNum = len(model)
		for i in range(classNum):
			joblib.dump(model[i], dirName+fileName+modelName + '_' + str(i) +'.pkl')
	else: # SVM, NN, Perceptron
		joblib.dump(model, dirName+fileName+modelName+'.pkl')




	




