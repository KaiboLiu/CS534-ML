import numpy as np
import matplotlib.pyplot as plt
import time
import pdb

import ClassifyModels as cf


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
	classLabel = [0, 1] # 0-Chinese, 1-English

	pdb.set_trace()
	gmmBag = cf.gmm_train(cfModel.trainData, classLabel)
	[gmmTest, gmmAcc] = cf.gmm_classify(cfModel.testData, gmmBag)
	print "GMM test result:\n", gmmAcc, gmmTest

	svmModel = cf.svm_train(cfModel.trainData)
	[svmTest, svmAcc] = cf.svm_classify(cfModel.testData, svmModel)
	print "\nsvm test result:\n", "accuracy: ", svmAcc, "\n test rst: ", svmTest

if __name__ == "__main__":
	RunMain()


