import numpy as np
from math import log

class NAIVE_BAYES_MODEL:
	""" class NAIVE_BAYES_MODEL, 
			1. this model is used for classification 
	        2. learn P(y_k) and P(x_i|y_k) for given model.
	        3. predict class for input X based on learning result.
	        4. support Bernoulli & Multinomial model
	        5. set Laplassian smooth for MAP estimation

		Functions: 
	"""
	def __init__(self, trainX = [], trainY = [], classNum = 2):
		self.trainX     = trainX
		self.trainY     = trainY

		self.classNum      = classNum
		self.sampleNum     = self.trainX.shape(1)
		self.featureNum    = self.trainX.shape(2)

		# prior distribution of y, And the distribution of X given y.
		self.Pwy_c      = []  # Pwy[i,k] = statistics count of word (w_i) given class (y_k). interger
		self.Py_c       = []  # Py[k]  = statistics sample count of class (y_k)
		self.PwyNorm_p  = []  # normalized probability of w_i given y_k, based on model & normalize method float
		self.Py_p       = []  # Py[i] = probability of specific class y_i, float

	def setTrainData(trainX, trainY):
		self.trainX     = trainX
		self.trainY     = trainY
		self.sampleNum  = self.trainX.shape(1)
		self.featureNum = self.trainX.shape(2)

	# estimate Py(the prior probability) using MLE (Maximum Likelihood Estimation). 	
	def estimatePy_MLE():
		# empty Py first, then estimate Py.
		del self.Py_p[:]
		for i in range(self.classNum):
			yi_cnt  = self.trainY.count(i)
			self.Py_c.append(yi_cnt)
			self.Py_p.append(yi_cnt/self.sampleNum)

	# convert a document record under multinomial model to that in bernoulli model.
	def vecXtranform_bernoulli(vecX):
		vecX_bool = vecX > 0 # bernoulli only consider if the word appear.
		vecX_01   = vecX_bool.astype(float) # boolean to float

		return vecX_01

	# based on training dataset, learn Pwy (count the number of word (w_i) appears given class (y_k))
	def estimatePwy_bernoulli():
		# empty Pwy first, then estimate Pwy.
		del self.Pwy_c[:]
		for k in range(self.classNum):
			y_idx = np.where(np.array(self.trainY) == k) # find all samples y_k

			Pw_yk = np.zeros(self.featureNum)
			for sj in y_idx:   #sj means sample j
				trainX_sj     = vecXtranform_bernoulli(trainX[sj,:])
				Pw_yk         = Pw_yk + bnl_trainX_sj

			self.Pwy_c.append(Pw_yk)

	# in order to avoid running into underflow issues,
	# laplace smooth of Pwy ( probability of word (w_i) appears given class (y_k)).
	def laplSmoothPwy_bernouli(lapAlpha = 1):
		denomi_lap = self.classNum * lapAlpha
		numer_lap  = np.ones(slef.featureNum)*lapAlpha
		for k in range(self.classNum):
			Pw_yk    = (self.Pwy_c[k,:] + numer_lap) / (self.Py_c[k] + denomi_lap)
			logPw_yk = np.log(Pw_yk)
			self.PwyNorm_p.append(logPw_yk)

	# vecX is the document record under bernoulli model,
	def calculatePx_bernoulli(vecX) 
		fea_one    = np.ones(self.featureNum)
		PvecX_y    = []
		for k in range(self.classNum):
			PvecX_pos = np.sum(self.PwyNorm_p[k,:]*vecX)
			PvecX_neg = np.sum((fea_one-self.PwyNorm_p[k,:])*(fea_one-vecX))
			PvecX_yk  = (PvecX_pos + PvecX_neg) * self.Py_p[k]
			PvecX_y.append(PvecX_yk)

		return np.array(PvecX_y)

	# predict class y give document vecX, return class Y that has the maximum probability
	def predictY_bernoulli(vecX):
		vecX     = vecXtranform_bernoulli(vecX)
		PvecX_y  = calculatePx_bernoulli(vecX)
		maxClass = np.where(PvecX_y == PvecX_y.max())

		return maxClass
	

	# based on training dataset, learn Pwy (count the number of word (w_i) appears given class (y_k))
	def estimatePxy_multinomial():
		# empty Pwy first, then estimate Pwy.
		del self.Pwy_c[:]
		for k in range(self.classNum):
			y_idx = np.where(np.array(self.trainY) == k) # find all samples y_k

			Pw_yk = np.zeros(self.featureNum)
			for sj in y_idx:   #sj means sample j
				trainX_sj     = trainX[sj,:]
				Pw_yk         = Pw_yk + np.array(trainX_sj)

			self.Pwy_c.append(Pw_yk)

	# in order to avoid running into underflow issues,
	# laplace smooth of Pwy ( probability of word (w_i) appears given class (y_k)).
	def laplSmoothPxy_multinomial(lapAlpha = 1):
		denomi_lap = self.featureNum * lapAlpha
		numer_lap  = np.ones(slef.featureNum)*lapAlpha
		for k in range(self.classNum):
			Pw_yk    = (self.Pwy_c[k,:] + numer_lap) / (self.featureNum + denomi_lap)
			logPw_yk = np.log(Pw_yk)
			self.PwyNorm_p.append(logPw_yk)

	# vecX is the document record under multinomial model
	def calculatePx_multinomial(vecX):
		PvecX_y    = []
		for k in range(self.classNum):
			PvecX_yk = np.dot(self.PwyNorm_p^T, vecX) * self.Py_p[k]
			PvecX_y.append(PvecX_yk)

		return np.array(PvecX_y)

	# predict class y give document x, return class Y that has the maximum probability
	def predictY_multinomial(vecX):
		PvecX_y  = calculatePx_multinomial(vecX)
		maxClass = np.where(PvecX_y == PvecX_y.max())

		return maxClass
				
	








