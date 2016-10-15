import numpy as np
import matplotlib.pyplot as plt
import time
import pdb

import LoadData   as ld
import NaiveBayes as nb

def LearnAndTest(naiveBayesModel, testX, testY, modelStr, lapAlpha = 1):
	testY_hat    = []
	testAccuracy = 0
	if(modelStr == "Bernoulli"):
		# learn Pwy based on Bernoulli model
		naiveBayesModel.estimatePwy_bernoulli()
		naiveBayesModel.laplSmoothPwy_bernouli(lapAlpha)

		# test on based on Py & Pwy
		for idx, doc in enumerate(testX):
			y_hat = naiveBayesModel.predictY_bernoulli(doc)
			testY_hat.append(y_hat)
			if (y_hat == testY[idx]):
				testAccuracy = testAccuracy + 1
	else: # learn Pwy based on Multinomial model
		naiveBayesModel.estimatePwy_multinomial()
		naiveBayesModel.laplSmoothPwy_multinomial(lapAlpha)

		# test on based on Py & Pwy
		for idx, doc in enumerate(testX):
			y_hat = naiveBayesModel.predictY_multinomial(doc)
			testY_hat.append(y_hat)
			if (y_hat == testY[idx]):
				testAccuracy = testAccuracy + 1

	return testAccuracy, testY_hat

def RunMain():
	print '************Welcome to the World of Bayes!***********\n'
	time.clock()
	t0 = float(time.clock())

	# # load data, and save as the format under NaiveBayes.
	DIR                   = "./clintontrump-data/"
	FILENAME_BASIC        = "clintontrump."
	wordNum               = ld.LoadData_vocabulary(DIR+FILENAME_BASIC+"vocabulary")
	[trainX, trainDocNum] = ld.LoadData_bagOfWords(DIR+FILENAME_BASIC+"bagofwords.train")
	[testX,  testDocNum]  = ld.LoadData_bagOfWords(DIR+FILENAME_BASIC+"bagofwords.dev")

	str0    = "realDonaldTrump"
	str1    = "HillaryClinton"
	trainY  = ld.LoadData_labels(DIR+FILENAME_BASIC+"labels.train", str0)
	testY   = ld.LoadData_labels(DIR+FILENAME_BASIC+"labels.dev", str0)
	t1 = float(time.clock())
	print 'Loading data File. using time %.4f s, \n' % (t1-t0)

	# # define NaiveBayes instance, and calc prior P(y)
	nbModel = nb.NAIVE_BAYES_MODEL(wordNum, trainDocNum, trainX, trainY)
	nbModel.estimatePy_MLE()

	# *******part 1: basic implementation
	###### Bernoulli model
	[berAccuracy, berTestHist] = LearnAndTest(nbModel, testX, testY, "Bernoulli")
	print 'Bernoulli accuracy is %.4f \n' %(float(berAccuracy)/float(testDocNum))
	t2 = float(time.clock())
	print 'Bernoulli Model learn & test, using time %.4f s, \n' % (t2-t1)

	###### Multinomial will go through the similar process.
	[mulAccuracy, mulTestHist] = LearnAndTest(nbModel, testX, testY, "Multinomial")
	print 'Multinomial accuracy is %.4f \n' %(float(mulAccuracy)/float(testDocNum))
	
	t3 = float(time.clock())
	print 'multinomial Model learn & test, using time %.4f s, \n' % (t3-t2)

	##### Ranking Top ten features
	# labelVec = [1, 0, 0, 1, ..., 0, 0, 1] 
	# redFeaNum = 10
	# nbModel.setFeatureLabel(labelVec, redFeaNum)
	# [berA, berHist] = LearnAndTest(nbModel, testX, testY, "Bernoulli")


	# ******** part 2: Priors and overfittings
	#  testAlpha = [...]
	# for each Alpha
	#     ...go through the similar process.


	# ******* part 3: bonus

	

if __name__ == "__main__":
	RunMain()