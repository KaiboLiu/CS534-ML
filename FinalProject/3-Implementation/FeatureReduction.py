import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification

def LoadFeature():
	dataDir    = "./Feature/"
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
    plt.show()

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
	#print "ReducedF:", selectedFName.keys()
	#print "\nReduced Feature Index:\n", selectedFName.values()
	return selectedFName.keys()

def find_vectors(trainX, new_trainX):
	# [12, 24, 36, 164, 184, 185, 186, 187, 194, 195, 197, 203, 204]
	selectedF = []
	for i in range(new_trainX.shape[1]):
		for j in range(trainX.shape[1]):
			if hamming_dist(new_trainX[:,i], trainX[:,j]) == 0:
				selectedF.append(j)
	featureName = match_with_features(np.array(selectedF))
	print "\nReduced Feature Index:\n", selectedF
	return featureName

def feature_selection_var(trainX, trainY, testX, testY, t=0.8):
	# Removing features with low variance
	sel = VarianceThreshold(threshold=(t * (1 - t)))
	new_trainX = sel.fit_transform(trainX)
	new_testX = sel.fit_transform(testX)
	print new_trainX.shape
	[x, y] = new_trainX.shape
	featureName = find_vectors(trainX, new_trainX)
	#print new_trainX, new_testX, y, featureName
	return new_trainX, new_testX, y, featureName

def feature_selection_L1(trainX, trainY, testX, testY, c=0.1):
	# L1-based feature selection
	lsvc = LinearSVC(C=c, penalty="l1", dual=False).fit(trainX, trainY)
	model = SelectFromModel(lsvc, prefit=True)
	new_trainX = model.transform(trainX)
	new_testX = model.transform(testX)
	print new_trainX.shape
	[x, y] = new_trainX.shape
	featureName = find_vectors(trainX, new_trainX)
	return new_trainX, new_testX, y, featureName

def feature_selection_tree(trainX, trainY, testX, testY):
	'''
	# Univariate feature selection
	#new_train = SelectKBest(chi2, k=2).fit_transform(trainX, trainY)
	#print new_train.shape
	'''

	# Tree-based feature selection
	clf = ExtraTreesClassifier()
	clf = clf.fit(trainX, trainY)
	clf.feature_importances_
	model = SelectFromModel(clf, prefit=True)
	new_trainX = model.transform(trainX)
	new_testX = model.transform(testX)
	print new_trainX.shape
	[x, y] = new_trainX.shape
	featureName = find_vectors(trainX, new_trainX)
	return new_trainX, new_testX, y, featureName

def important_features_from_tree():
	# Build a classification task using 3 informative features
	train = np.genfromtxt("./Feature/train.dev")
	[N, dim] = train.shape
	y = train[:, dim-1]
	X = np.delete(train, 1, 1)

	# Build a forest and compute the feature importances
	forest = ExtraTreesClassifier(n_estimators=250,
	                              random_state=0)

	forest.fit(X, y)
	importances = forest.feature_importances_
	print importances
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1][:3]
	print indices.shape[0]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(indices.shape[0]):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(indices.shape[0]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), indices)
	plt.xlim([-1, indices.shape[0]])
	plt.show()

def feature_selection_lda(trainX, trainY, testX, testY):
	new_trainX = LDA_analysis(trainX, trainY)
	new_testX = LDA_analysis(testX, testY)
	print new_trainX.shape
	featureName = []
	return new_trainX, new_testX, 1, featureName

def RunMain():
	[trainX, trainY, testX, testY] = LoadFeature()
	LDA_analysis(trainX, trainY)
	feature_selection_var(trainX, trainY, testX, testY, t=0.8)
	feature_selection_L1(trainX, trainY, testX, testY, c=0.1)
	feature_selection_tree(trainX, trainY, testX, testY)
	feature_selection_lda(trainX, trainY, testX, testY)


if __name__ == "__main__":
	RunMain()
