import numpy as np
import matplotlib.pyplot as plt
import time
import pdb

import LoadData   as ld
import OutputData as od
import NaiveBayes as nb

def LearnAndTest(naiveBayesModel, testX, testY, modelStr, lapAlpha = 1):
	confuseMat   = []
	confuseMat.append([0,0])
	confuseMat.append([0,0])

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
			confuseMat[y_hat][testY[idx]] = confuseMat[y_hat][testY[idx]] + 1
			#if (y_hat == testY[idx]):
				#testAccuracy = testAccuracy + 1
	else: # learn Pwy based on Multinomial model
		naiveBayesModel.estimatePwy_multinomial()
		naiveBayesModel.laplSmoothPwy_multinomial(lapAlpha)

		# test on based on Py & Pwy
		for idx, doc in enumerate(testX):
			y_hat = naiveBayesModel.predictY_multinomial(doc)
			testY_hat.append(y_hat)
			confuseMat[y_hat][testY[idx]] = confuseMat[y_hat][testY[idx]] + 1
			#if (y_hat == testY[idx]):
				#testAccuracy = testAccuracy + 1
	testAccuracy = confuseMat[0][0]+confuseMat[1][1]
	return testAccuracy, testY_hat, confuseMat
'''
def PriorAndFitting_diffLaplace(naiveBayesModel, , testX, testY):
	testY_alpha = []
	for alpha in testY_alpha:
		[accuracy, testHist, confuseMat] = LearnAndTest(nbModel, testX, testY, "Multinomial", alpha)
		od.WritenFile_dev(DIR+"Predict.Multinomial"+str(alpha)=".dev", berTestHist, str0, str1)
		print 'Multinomial accuracy is %.4f confuseMatrix is:\n' %(float(berAccuracy)/float(testDocNum)), berConfuseMat
	


	return testAlpha, testAccuracy
'''
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
	[berAccuracy, berTestHist, berConfuseMat] = LearnAndTest(nbModel, testX, testY, "Bernoulli")
	od.WritenFile_dev(DIR+"Predict.Bernoulli_0.dev", berTestHist, str0, str1)
	print 'Bernoulli accuracy is %.4f confuseMatrix is:\n' %(float(berAccuracy)/float(testDocNum)), berConfuseMat
	t2 = float(time.clock())
	print 'Bernoulli Model learn & test, using time %.4f s, \n' % (t2-t1)

	###### Multinomial will go through the similar process.
	#pdb.set_trace()
	[mulAccuracy, mulTestHist, mulConfuseMat] = LearnAndTest(nbModel, testX, testY, "Multinomial", 1)
	od.WritenFile_dev(DIR+"Predict.Multinomial_0.dev", mulTestHist, str0, str1)
	print 'Multinomial accuracy is %.4f \n confuse matrix is:\n' %(float(mulAccuracy)/float(testDocNum)), mulConfuseMat
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