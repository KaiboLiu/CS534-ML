import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import copy

import LoadData   as ld
import OutputData as od
import NaiveBayes as nb
import Improvement as im

def LearnAndTest(naiveBayesModel, testX, testY, modelStr, lapAlpha = 1):
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
			y_hat = naiveBayesModel.predictY_bernoulli(doc)
			testY_hat.append(y_hat)
			confuseMat[y_hat][testY[idx]] = confuseMat[y_hat][testY[idx]] + 1
		
		testAccuracy = confuseMat[0][0]+confuseMat[1][1]
		print 'Bernoulli accuracy is %.4f \nconfuseMatrix is:\n' %(float(testAccuracy)/float(devDocNum)), confuseMat

	else: # learn Pwy based on Multinomial model
		naiveBayesModel.estimatePwy_multinomial()
		naiveBayesModel.laplSmoothPwy_multinomial(lapAlpha)

		# test on based on Py & Pwy
		for idx, doc in enumerate(testX):
			y_hat = naiveBayesModel.predictY_multinomial(doc)
			testY_hat.append(y_hat)
			confuseMat[y_hat][testY[idx]] = confuseMat[y_hat][testY[idx]] + 1
		
		testAccuracy = confuseMat[0][0]+confuseMat[1][1]
		print 'Multinomial accuracy is %.4f \nconfuseMatrix is:\n' %(float(testAccuracy)/float(devDocNum)), confuseMat

	#testAccuracy = confuseMat[0][0]+confuseMat[1][1]
	return testAccuracy, testY_hat, confuseMat

def PriorAndFitting_diffLaplace(naiveBayesModel, testX, testY, dir, str0, str1):
	print '\n\n****** Now we are checking different laplace smooth alpha\n'
	testY_alpha  = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
	testAccuracy = []
	for alpha in testY_alpha:
		filename = dir+"Predict.Mul.diffLaps."+str(alpha)+".dev"
		[accuracy, testHist, confuseMat] = LearnAndTest(naiveBayesModel, testX, testY, "Multinomial", alpha)
		testAccuracy.append(accuracy)
		od.WritenFile_dev(filename, testHist, str0, str1)

	return testY_alpha, testAccuracy

def showWords(vocList, idx_list):
	if type(idx_list[0]) is list:
		for k in range(len(idx_list)):
			print "idx: ", idx_list[k]
			print "Words: ", vocList[idx_list[k]]
	else:
		print "idx: ", idx_list
		print "Words: ", vocList[idx_list]

def RunMain():
	print '************Welcome to the World of Bayes!***********\n'
	time.clock()
	t0 = float(time.clock())

	# # load data, and save as the format under NaiveBayes.
	DIR_RESULT            = "./Result/"
	DIR                   = "./clintontrump-data/"
	FILENAME_BASIC        = "clintontrump."
	[vocList, wordNum]    = ld.LoadData_vocabulary(DIR+FILENAME_BASIC+"vocabulary")
	[trainX, trainDocNum] = ld.LoadData_bagOfWords(DIR+FILENAME_BASIC+"bagofwords.train")
	[devX,  devDocNum]    = ld.LoadData_bagOfWords(DIR+FILENAME_BASIC+"bagofwords.dev")
	[testX,  testDocNum]  = ld.LoadData_bagOfWords(DIR+FILENAME_BASIC+"bagofwords.test")

	str0    = "realDonaldTrump"
	str1    = "HillaryClinton"
	trainY  = ld.LoadData_labels(DIR+FILENAME_BASIC+"labels.train", str0)
	devY   = ld.LoadData_labels(DIR+FILENAME_BASIC+"labels.dev", str0)
	t1 = float(time.clock())
	print 'Loading data File. using time %.4f s, \n' % (t1-t0)

	# # define NaiveBayes instance, and calc prior P(y)
	nbModel = nb.NAIVE_BAYES_MODEL(wordNum, trainDocNum, trainX, trainY)
	nbModel.estimatePy_MLE()

	# *******part 1: basic implementation
	###### Bernoulli model
	[berAccuracy, berTestHist, berConfuseMat] = LearnAndTest(nbModel, devX, devY, "Bernoulli")
	od.WritenFile_dev(DIR_RESULT+"Predict.Bernoulli_basic.dev", berTestHist, str0, str1)
	Pwy_b = copy.deepcopy(nbModel.Pwy_c)
	#print 'Bernoulli accuracy is %.4f \nconfuseMatrix is:\n' %(float(berAccuracy)/float(testDocNum)), berConfuseMat
	t2 = float(time.clock())
	print 'Bernoulli Model learn & test, using time %.4f s, \n' % (t2-t1)

	###### Multinomial will go through the similar process.
	[mulAccuracy, mulTestHist, mulConfuseMat] = LearnAndTest(nbModel, devX, devY, "Multinomial")
	od.WritenFile_dev(DIR_RESULT+"Predict.Multinomial_basic.dev", mulTestHist, str0, str1)
	Pwy_m = copy.deepcopy(nbModel.Pwy_c)
	#print 'Multinomial accuracy is %.4f \nconfuse matrix is:\n' %(float(mulAccuracy)/float(testDocNum)), mulConfuseMat
	t3 = float(time.clock())
	print 'multinomial Model learn & test, using time %.4f s, \n' % (t3-t2)

	'''
	##### Ranking Top ten features
	topWord_num = 1000 #[10, 100, 1000, 5000]
	std_threshold = 0.5 #[0, 0.5, 1, 2]
	tfidf_threshold = 2 #[1, 2, 3, 4]
	print '\n * Remove top words'
	[removedIdx, labelVec, redFeaNum] = im.find_top_words(nbModel.classNum, wordNum, topWord_num, Pwy_b)
	showWords(vocList, removedIdx)
	nbModel.setFeatureLabel(labelVec, redFeaNum)
	[berAccuracy, berTestHist, berConfuseMat] = LearnAndTest(nbModel, devX, devY, "Bernoulli")
	[mulAccuracy, mulTestHist, mulConfuseMat] = LearnAndTest(nbModel, devX, devY, "Multinomial")

	# Find stop words based on STD and doc#
	print '\n * Remove STD words for Bernoulli'
	[removedIdx, labelVec, redFeaNum] = im.find_std_zero_words(wordNum, std_threshold, Pwy_b)
	nbModel.setFeatureLabel(labelVec, redFeaNum)
	[berAccuracy, berTestHist, berConfuseMat]  = LearnAndTest(nbModel, devX, devY, "Bernoulli")
	[mulAccuracy, mulTestHist, mulConfuseMat] = LearnAndTest(nbModel, devX, devY, "Multinomial")

	# Find stop words based on STD and word#
	print '\n * Remove STD words for Multinomial'
	[removedIdx, labelVec, redFeaNum] = im.find_std_zero_words(wordNum, std_threshold, Pwy_m)
	nbModel.setFeatureLabel(labelVec, redFeaNum)
	[berAccuracy, berTestHist, berConfuseMat]  = LearnAndTest(nbModel, devX, devY, "Bernoulli")
	[mulAccuracy, mulTestHist, mulConfuseMat] = LearnAndTest(nbModel, devX, devY, "Multinomial")

	# Find stop words based on TF/IDF
	print '\n * Remove low TF/IDF words'
	[removedIdx, labelVec, redFeaNum] = im.find_low_tfidf_words(wordNum, tfidf_threshold, Pwy_b, Pwy_m)
	nbModel.setFeatureLabel(labelVec, redFeaNum)
	[berAccuracy, berTestHist, berConfuseMat]  = LearnAndTest(nbModel, devX, devY, "Bernoulli")
	[mulAccuracy, mulTestHist, mulConfuseMat] = LearnAndTest(nbModel, devX, devY, "Multinomial")
	'''
	
	# ******** part 2: Priors and overfittings
	## different Laplace Smoothing Alpha
	[testAlpha, testAccuracy] = PriorAndFitting_diffLaplace(nbModel, devX, devY, DIR_RESULT, str0, str1)
	testAccuracy = np.array(testAccuracy)/float(testDocNum)
	print testAlpha
	print testAccuracy
	od.Save2Figure_semilogs(DIR_RESULT+'laplaceAlpha', 1, testAlpha, [testAccuracy],['log(laplace_alpha)','accuracy'], [1e-5, 10000,0, 1], 1)
	
	

	# ******* part 3: bonus


if __name__ == "__main__":
	RunMain()