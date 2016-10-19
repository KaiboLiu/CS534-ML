import numpy as np
from math import log
import pdb

class NAIVE_BAYES_MODEL:
	""" class NAIVE_BAYES_MODEL,
			1. this model is used for classification
	        2. learn P(y_k) and P(x_i|y_k) for given model.
	        3. predict class for input X based on learning result.
	        4. support Bernoulli & Multinomial model
	        5. set Laplassian smooth for MAP estimation

		Functions:
	"""
	def __init__(self, wordNum, docNum, trainX = [], trainY = [], classNum = 2):
		self.trainX     = trainX
		self.trainY     = trainY

		self.classNum      = classNum
		self.sampleNum     = docNum
		self.featureNum    = wordNum
		print "featureNum:", wordNum
		self.featureLabel  = np.ones(self.featureNum)

		# prior distribution of y, And the distribution of X given y.
		self.Pwy_c        = []  # Pwy[i,k] = statistics count of word (w_i) given class (y_k). interger
		self.Py_c         = []  # Py[k]  = statistics sample count of class (y_k)
		self.PwyNorm_p    = []  # PwyNorm_p = log normalized probability of w_i given y_k, based on model & normalize method float
		self.PwyNorm_negP = []  # for Bernoulli model, the log normalized probability of no w_i given y_k
		self.Py_p         = []  # Py[i] = log probability of specific class y_i, float

	def setTrainData(self, trainX, trainY, wordNum, docNUm):
		self.trainX     = trainX
		self.trainY     = trainY
		self.sampleNum  = docNum
		self.featureNum = wordNum
		self.featureLabel = np.ones(self.featureNum)

	# after feature reduction, call this function to adjust Bayes dataset.
	def setFeatureLabel(self, labelVec, redFeatureNum):
		print "redFeatureNum:", redFeatureNum
		self.featureLabel = labelVec
		self.featureNum   = redFeatureNum

	# estimate Py(the prior probability) using MLE (Maximum Likelihood Estimation).
	def estimatePy_MLE(self):
		# empty Py first, then estimate Py.
		del self.Py_c[:]
		del self.Py_p[:]
		for i in range(self.classNum):
			yi_cnt  = self.trainY.count(i)
			self.Py_c.append(yi_cnt)
			self.Py_p.append(np.log(float(yi_cnt)/float(self.sampleNum)))

	# create a vector of X, contains information of all words.
	# input is a sparse vector, it's a document in a form of word index in vocabulary.
	def createVecX(self, sparseX):
		vecX  = np.zeros(self.featureNum)
		count = 0
		for idx, label in enumerate(self.featureLabel):
			if label == 1:
				vecX[count] = sparseX.count(idx)
				count = count + 1

		return vecX

	# convert a document record under multinomial model to that in bernoulli model.
	def vecXtranform_bernoulli(self, vecX):
		vecX_bool = vecX > 0 # bernoulli only consider if the word appear.
		vecX_01   = vecX_bool.astype(float) # boolean to float

		return vecX_01

	# based on training dataset, learn Pwy (count the number of word (w_i) appears given class (y_k))
	def estimatePwy_bernoulli(self):
		# empty Pwy first, then estimate Pwy.
		del self.Pwy_c[:]
		for yk in range(self.classNum):
			# yk_idx = np.where(np.array(self.trainY) == yk)
			Pw_yk  = np.zeros(self.featureNum)
			Pw_yk = np.zeros(self.featureNum)
			for sj, classY in enumerate(self.trainY):
				if(classY != yk):
					continue
				trainX_vec    = self.createVecX(self.trainX[sj])
				trainX_sj     = self.vecXtranform_bernoulli(trainX_vec)
				Pw_yk         = Pw_yk + trainX_sj

			self.Pwy_c.append(Pw_yk)

	# in order to avoid running into underflow issues,
	# laplace smooth of Pwy ( probability of word (w_i) appears given class (y_k)).
	def laplSmoothPwy_bernouli(self, lapAlpha = 1):
		del self.PwyNorm_p[:]
		del self.PwyNorm_negP[:]
		denomi_lap = self.classNum * lapAlpha
		numer_lap  = lapAlpha
		for k in range(self.classNum):
			Pw_yk    = (np.array(self.Pwy_c[k]) + numer_lap) / float(self.Py_c[k] + denomi_lap)
			logPw_yk = np.log(Pw_yk)
			logPnw_yk = np.log(1 - Pw_yk)
			self.PwyNorm_p.append(logPw_yk.tolist())
			self.PwyNorm_negP.append(logPnw_yk.tolist())

	# vecX is the document record under bernoulli model,
	def calculatePx_bernoulli(self, vecX):
		fea_one    = np.ones(self.featureNum)
		PvecX_y    = []
		for k in range(self.classNum):
			PvecX_pos = np.sum(self.PwyNorm_p[k]*vecX)
			PvecX_neg = np.sum(self.PwyNorm_negP[k]*(fea_one-vecX))
			PvecX_yk  = (PvecX_pos + PvecX_neg) + self.Py_p[k]
			PvecX_y.append(PvecX_yk)

		return np.array(PvecX_y)

	# predict class y give document vecX, return class Y that has the maximum probability
	def predictY_bernoulli(self, docX):
		vecX     = self.createVecX(docX)
		vecX     = self.vecXtranform_bernoulli(vecX)
		PvecX_y  = self.calculatePx_bernoulli(vecX)
		maxClass = np.argmax(PvecX_y)

		return maxClass

	# based on training dataset, learn Pwy (count the number of word (w_i) appears given class (y_k))
	def estimatePwy_multinomial(self):
		# empty Pwy first, then estimate Pwy.
		del self.Pwy_c[:]
		for yk in range(self.classNum):
			Pw_yk = np.zeros(self.featureNum)
			for sj, classY in enumerate(self.trainY):
				if(yk != classY):
					continue
				trainX_sj  = self.createVecX(self.trainX[sj])
				Pw_yk      = Pw_yk + np.array(trainX_sj)

			self.Pwy_c.append(Pw_yk)

	# in order to avoid running into underflow issues,
	# laplace smooth of Pwy ( probability of word (w_i) appears given class (y_k)).
	def laplSmoothPwy_multinomial(self, lapAlpha = 1):
		del self.PwyNorm_p[:]
		del self.PwyNorm_negP[:]
		denomi_lap = self.featureNum * lapAlpha
		numer_lap  = lapAlpha
		for k in range(self.classNum):
			Pw_yk    = (np.array(self.Pwy_c[k]) + numer_lap) / float(self.featureNum + denomi_lap)
			logPw_yk = np.log(Pw_yk)
			self.PwyNorm_p.append(logPw_yk.tolist())
	# vecX is the document record under multinomial model
	def calculatePx_multinomial(self, vecX):
		PvecX_y    = []
		for k in range(self.classNum):
			PvecX_yk = np.sum(self.PwyNorm_p[k]*vecX) + self.Py_p[k]
			PvecX_y.append(PvecX_yk)

		return np.array(PvecX_y)

	# predict class y give document x, return class Y that has the maximum probability
	def predictY_multinomial(self, docX):
		vecX     = self.createVecX(docX)
		PvecX_y  = self.calculatePx_multinomial(vecX)
		maxClass = np.argmax(PvecX_y)

		return maxClass
  
	'''
	the codes below is for tag marked optimazation for predictions
 	'''
	# predict class y give document vecX, return class Y that has the maximum probability
	def predictY_bernoulli_withtag(self, docX):
		tag = set([5222,9171,12567,7295,8186,10438,11428,])
		if (len(set(docX).intersection(tag)) > 0):
			return 0
		vecX     = self.createVecX(docX)
		vecX     = self.vecXtranform_bernoulli(vecX)
		PvecX_y  = self.calculatePx_bernoulli(vecX)
		maxClass = np.argmax(PvecX_y)

		return maxClass
  
	# predict class y give document x, return class Y that has the maximum probability
	def predictY_multinomial_withtag(self, docX):
		tag = set([5222,9171,12567,7295,8186,10438,11428,])
		if (len(set(docX).intersection(tag)) > 0):
			return 0
		vecX     = self.createVecX(docX)
		PvecX_y  = self.calculatePx_multinomial(vecX)
		maxClass = np.argmax(PvecX_y)

		return maxClass