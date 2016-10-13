import numpy as np
import matplotlib.pyplot as plt

import NaiveBayes as nb

'''
def AccuracyTest(testX, testY, naiveBayesModel, str_model):
	# testY_hat   = []
	# testAccuracy = 0 
	# for i in range(testX.shape(1)):
	#     if str_model == 'Bernoulli':
	#     	y_hat = nbModel.predictY_bernoulli(testX[i,:])
	#	  else
	#		y_hat = nbModel.predictY_bernoulli(textX[i,:])
	#
	#     testY_hat.append(y_hat)
	#     if (y_hat == testY[i])
	# 			testY
	return testAccuracy
'''

def RunMain(val):
	print 'Yor are in RunMain\n'

	# # load data, and save as the format under NaiveBayes.
	# [trainX, trainY] = LoadData('train')
	# [testX, textY] = LoadData('test')

	# # define NaiveBayes instance, and calc prior P(y)
	# nbModel = nb.NAIVE_BAYES_MODEL(trainX, trainY)
	# nbModel.estimatePy_MLE()

	# *******part 1: basic implementation
	# # # # #Bernoulli model:
	# # learn Pwy (probability of word given y)
	# nbModel.estimatePwy_bernoulli()
	# nbModel.laplSmoothPwy_bernouli()
	# berAccuracy = AccuracyTest(testX, testY, nbModel, 'Bernoulli')

	###### Multinomial will go through the similar process.


	# ******** part 2: Priors and overfittings
	#  testAlpha = [...]
	# for each Alpha
	#     ...go through the similar process.


	# ******* part 3: bonus

	

if __name__ == "__main__":
	RunMain(3)