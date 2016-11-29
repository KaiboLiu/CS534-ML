import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier


def LoadFeature():
	dataDir    = "./Data/Feature/"
	trainFeature  = "train.dev"
	testFeature   = "test.dev"

	train = np.genfromtxt(dataDir+trainFeature)
	[N, dim] = train.shape
	trainY = train[:, dim-1]
	trainX = np.delete(train, 1, 1)

	test = np.genfromtxt(dataDir+testFeature)
	testY = test[:, dim-1]
	testX = np.delete(test, 1, 1)
	return trainX, trainY, testX, testY

def computeMeanVar(Data, Label, class_label):
    members = Data[np.where(Label == class_label)]
    mean = np.mean(members, 0)
    scatter = np.dot((members - mean).T, (members - mean))
    return mean, scatter, members

def LDA_analysis(trainX, trainY):
    [mean0, scatter0, members0] = computeMeanVar(trainX, trainY, 0)
    [mean1, scatter1, members1] = computeMeanVar(trainX, trainY, 1)
    S = scatter0 + scatter1
    S = scatter0 + scatter1
    if abs(det(S)) < 1e-5:
        S += 1e-4*np.eye(S.shape[1])
    w = np.dot(inv(S), (mean0 - mean1))
    w_norm = w / norm(w)
    projected_members0 = np.dot(members0, w_norm)
    projected_members1 = np.dot(members1, w_norm)

    plt.figure(2)
    plt.hist(projected_members0, color='r', alpha=0.5, label='class 0')
    plt.hist(projected_members1, color='b', alpha=0.5, label='class 1')
    plt.xlabel('projected data values')
    plt.ylabel('frequency')
    plt.legend(loc='upper right')
    #plt.show()

    return np.dot(trainX, w_norm)

def hamming_dist(x1, x2):
	return np.sum(np.abs(x1-x2))

def match_with_features(a):
	featureName = ['chromagram', 'Qchromagram', 'cens', 'melspec',
	'mfcc', 'rmse', 'centroid', 'bandwidth', 'contrast', 'rolloff',
	'poly_features', 'tonnetz', 'zero_crossing_rate']
	featureIdx = [12,24,36,164,184,185,186,187,194,195,197,203,204]

	selectedFName = {}
	pre_i = 0
	for i in range(len(featureName)):
		idx = np.where(np.logical_and(a>=pre_i, a<featureIdx[i]))
		if idx[0].size > 0:
			selectedFName[featureName[i]] = idx
		pre_i = featureIdx[i]
	print "ReducedF:", selectedFName.keys()
	#print "\nReduced Feature Index:\n", selectedFName.values()

def find_vectors(trainX, new_trainX):
	# [12, 24, 36, 164, 184, 185, 186, 187, 194, 195, 197, 203, 204]
	selectedF = []
	for i in range(new_trainX.shape[1]):
		for j in range(trainX.shape[1]):
			if hamming_dist(new_trainX[:,i], trainX[:,j]) == 0:
				selectedF.append(j)
	match_with_features(np.array(selectedF))
	#print "\nReduced Feature Index:\n", selectedF

def feature_selection(trainX, trainY):
	# Removing features with low variance
	t = 0.5
	sel = VarianceThreshold(threshold=(t * (1 - t)))
	new_trainX = sel.fit_transform(trainX)
	print new_trainX.shape
	find_vectors(trainX, new_trainX)

	# Univariate feature selection
	#new_train = SelectKBest(chi2, k=2).fit_transform(trainX, trainY)
	#print new_train.shape

	# L1-based feature selection
	lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(trainX, trainY)
	model = SelectFromModel(lsvc, prefit=True)
	new_trainX = model.transform(trainX)
	print new_trainX.shape
	find_vectors(trainX, new_trainX)

	# Tree-based feature selection
	clf = ExtraTreesClassifier()
	clf = clf.fit(trainX, trainY)
	clf.feature_importances_
	model = SelectFromModel(clf, prefit=True)
	new_trainX = model.transform(trainX)
	print new_trainX.shape
	find_vectors(trainX, new_trainX)

def RunMain():
	[trainX, trainY, testX, testY] = LoadFeature()
	LDA_analysis(trainX, trainY)
	feature_selection(trainX, trainY)

if __name__ == "__main__":
	RunMain()
