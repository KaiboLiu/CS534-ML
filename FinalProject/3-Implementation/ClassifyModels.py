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
import collections
import random
# import matplotlib.pyplot as plt
# import matplotlib as mpl


class ClassModel:
	def __init__(self, trainData=[], testData=[]):
		self.trainData = trainData
		self.testData  = testData
		self.norTrainData = []
		self.norTestData  = []

	def readFile(self, fileName, isTrain):
		if isTrain == True:
			self.trainData    = np.genfromtxt(fileName)
		else:
			self.testData    = np.genfromtxt(fileName)

	def divideTrainData(self, percent):
		train_size = len(self.trainData)
		sset_size  = np.int(train_size*percent)
		sset_idx   = random.sample(range(train_size), sset_size)
		train_sset = self.trainData[sset_idx[:]]

		val_idx  = set(range(train_size)) - set(sset_idx)
		val_sset = self.trainData[val_idx[:]]

		return train_sset, val_sset

	def featureNormalize(self):
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

	def neuralNetwork(self):
		# training data to fit model
		kernel = ['linear', 'rbf','poly']
		clf = svm.SVC(kernel[2])
		clf.fit(self.trainData[:,0:-1], self.trainData[:,-1])

		# test on testData
		accuracy = 0
		testResult = []
		for row in testData:
			testRst = clf.predict(row[0:-1])
			if testRst == row[-1]:
				accuracy += 1
			testResult.append(testRst)

		return testResult, accuracy


def gmm_train(trainData, classLabel, maxModelNum = 2):
	# statistic class number and examples.
	class_trainData = []
	classNum        = 0
	for k in classLabel:
		tmp = [ele[0:-1] for ele in trainData if ele[-1] == k]
		class_trainData.append(tmp)
		classNum = classNum + 1

	# GMM fit
	bestGMM  = []
	cv_types = ['spherical', 'tied', 'diag', 'full']
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

def gmm_classify(testData, modelBag):
	classNum = len(modelBag)
	
	# test on GMM model
	testResult = []
	accuracy = 0
	for row in testData:
		bestScore = -np.infty
		bestLabel = None
		for k in range(classNum):
			score = modelBag[k].score(row[0:-1].reshape(1,-1)) # how score looks like?
			if(score > bestScore):
				bestScore = score
				bestLabel = k
		if bestLabel == row[-1]:
			accuracy += 1
		testResult.append(bestLabel)
	return testResult, accuracy