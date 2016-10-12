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
		self.Pxy_c      = []  # Pxy[i,k] = statistics count of x_i given y_k. interger
		self.PxyNorm_p  = []  # normalized probability of x_i given y_k. float
		self.Py_p       = []  # Py[i] = probability of specific y_i, float

	def setTrainData(trainX, trainY):
		self.trainX     = trainX
		self.trainY     = trainY
		self.sampleNum  = self.trainX.shape(1)
		self.featureNum = self.trainX.shape(2)

	# estimate Py using MLE (Maximum Likelihood Estimation). 	
	def estimatePy_MLE():
		# empty Py first, then estimate Py.
		del self.Py_p[:]
		for i in range(self.classNum):
			yi_cnt  = self.trainY.count(i)
			Py_p.append(yi_cnt/self.sampleNum)

	def vecXtranform_bernoulli(vecX):
		vecX_bool = vecX > 0 # bernoulli only consider if the word appear.
		vecX_01   = vecX_bool.astype(float) # boolean to float

		return vecX_01

	# based on training dataset, learn Pxy using MLE or MAP(Maximum A Posteiror.)
	def estimatePxy_bernoulli():
		# empty Pxy first, then estimate Pxy.
		del self.Pxy_c[:]
		for k in range(self.classNum):
			y_idx = np.where(np.array(self.trainY) == k) # find all samples y_k

			Px_yk = np.zeros(self.featureNum)
			for sj in y_idx:   #sj means sample j
				trainX_sj     = vecXtranform_bernoulli(trainX[sj,:])
				Px_yk         = Px_yk + bnl_trainX_sj

			self.Pxy_c.append(Px_yk)

	# laplace smooth of Pxy, in order to avoid running into underflow issues.
	def laplSmoothPxy_bernouli(lapAlpha = 1):
		denomi_lap = self.classNum * lapAlpha
		numer_lap  = np.ones(slef.featureNum)*lapAlpha
		for k in range(self.classNum):
			Px_yk    = (self.Pxy[k,:] + numer_lap) / (self.Pxy[k,:].sum() + denomi_lap)
			logPx_yk = np.log(Px_yk)
			self.PxyNorm_p.append(logPx_yk)

	# vecX is the input Feature Vector,
	# lapAlpha is the factor for Laplace smoothing
	def calculatePx_bernoulli(vecX)
		fea_one    = np.ones(self.featureNum)
		PvecX_y    = []
		for k in range(self.classNum):
			PvecX_pos = np.sum(self.PxyNorm_p[k,:]*vecX)
			PvecX_neg = np.sum((fea_one-self.PxyNorm_p[k,:])*(fea_one-vecX))
			PvecX_yk  = (PvecX_pos + PvecX_neg) * self.Py_p[k]
			PvecX_y.append(PvecX_yk)

		return np.array(PvecX_y)

	# predict class y give x, return class Y that has the maximum probability
	def predictY_bernoulli(vecX):
		vecX     = vecXtranform_bernoulli(vecX)
		PvecX_y  = calculatePx_bernoulli(vecX)
		maxClass = np.where(PvecX_y == PvecX_y.max())

		return maxClass
	

	# based on training dataset, learn Pxy using MLE or MAP.
	def estimatePxy_multinomial():
		# empty Pxy first, then estimate Pxy.
		del self.Pxy[:]
		for k in range(self.classNum):
			y_idx = np.where(np.array(self.trainY) == k) # find all samples y_k

			Px_yk = np.zeros(self.featureNum)
			for sj in y_idx:   #sj means sample j
				trainX_sj     = trainX[sj,:]
				Px_yk         = Px_yk + np.array(trainX_sj)

			self.Pxy.append(Px_yk)

	######
	# laplace smooth of Pxy, in order to avoid running into underflow issues.
	def laplSmoothPxy_multinomial(lapAlpha = 1):
		denomi_lap = self.featureNum * lapAlpha
		numer_lap  = np.ones(slef.featureNum)*lapAlpha
		for k in range(self.classNum):
			Px_yk    = (self.Pxy[k,:] + numer_lap) / (self.Pxy[k,:].sum() + denomi_lap)
			logPx_yk = np.log(Px_yk)
			self.PxyNorm_p.append(logPx_yk)

	# vecX is the input Feature Vector,
	# lapAlpha is the factor for Laplace smoothing
	def calculatePx_multinomial(vecX):
		PvecX_y    = []
		for k in range(self.classNum):
			PvecX_yk = np.dot(Px_yk^T, vecX) * Pyk
			PvecX_y.append(PvecX_yk)

		return np.array(PvecX_y)

	# predict class y give x, return class Y that has the maximum probability
	def predictY_multinomial(vecX):
		PvecX_y  = calculatePx_multinomial(vecX)
		maxClass = np.where(PvecX_y == PvecX_y.max())

		return maxClass
				
	








