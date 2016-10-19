# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:13:15 2016

@author: Kaibo Liu
"""
import NaiveBayes as nb


def LearnAndTest(naiveBayesModel, testX, testY, modelStr,lapAlpha = 1):
	confuseMat   = []
	confuseMat.append([0,0])
	confuseMat.append([0,0])

	testY_hat    = []
	testAccuracy = 0
	devDocNum = len(testY)
	if(modelStr == "Bernoulli"):
		# learn Pwy based on Bernoulli model
		naiveBayesModel.estimatePwy_bernoulli()
		naiveBayesModel.laplSmoothPwy_bernouli(lapAlpha)

		# test on based on Py & Pwy
		for idx, doc in enumerate(testX):
			y_hat = naiveBayesModel.predictY_bernoulli_withtag(doc)
			testY_hat.append(y_hat)
			confuseMat[y_hat][testY[idx]] = confuseMat[y_hat][testY[idx]] + 1
			#if (y_hat != testY[idx]):
			#	print idx,testY[idx],y_hat, testX[idx]

		testAccuracy = confuseMat[0][0]+confuseMat[1][1]
		print 'Bernoulli accuracy after tag marked is %.4f \nconfuseMatrix is:\n' %(float(testAccuracy)/float(devDocNum)), confuseMat

	else: # learn Pwy based on Multinomial model
		naiveBayesModel.estimatePwy_multinomial()
		naiveBayesModel.laplSmoothPwy_multinomial(lapAlpha)

		# test on based on Py & Pwy
		for idx, doc in enumerate(testX):
			y_hat = naiveBayesModel.predictY_multinomial_withtag(doc)
			testY_hat.append(y_hat)
			confuseMat[y_hat][testY[idx]] = confuseMat[y_hat][testY[idx]] + 1

		testAccuracy = confuseMat[0][0]+confuseMat[1][1]
		print 'Multinomial accuracy after tag marked is %.4f \nconfuseMatrix is:\n' %(float(testAccuracy)/float(devDocNum)), confuseMat

	#testAccuracy = confuseMat[0][0]+confuseMat[1][1]
	return testAccuracy, testY_hat, confuseMat
 
 
def LearnAndPredict(naiveBayesModel, testX, modelStr,lapAlpha = 1):

	testY_hat    = []
	if(modelStr == "Bernoulli"):
		# learn Pwy based on Bernoulli model
		naiveBayesModel.estimatePwy_bernoulli()
		naiveBayesModel.laplSmoothPwy_bernouli(lapAlpha)

		# test on based on Py & Pwy
		for idx, doc in enumerate(testX):
			y_hat = naiveBayesModel.predictY_bernoulli_withtag(doc)
			testY_hat.append(y_hat)

	else: # learn Pwy based on Multinomial model
		naiveBayesModel.estimatePwy_multinomial()
		naiveBayesModel.laplSmoothPwy_multinomial(lapAlpha)

		# test on based on Py & Pwy
		for idx, doc in enumerate(testX):
			y_hat = naiveBayesModel.predictY_multinomial_withtag(doc)
			testY_hat.append(y_hat)
	return testY_hat
