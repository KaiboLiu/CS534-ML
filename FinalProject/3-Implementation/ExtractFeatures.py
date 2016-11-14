import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt

def Written2File(fileName, vecV):
	f = open(fileName, "w")
	for v in vecV:
		line = " ".join([str(vecV(fea)) for fea in v.split()])
		f.write(line + "\n")
	f.close()

def ExtreactMFCC(train, test):
	train_feature = np.zeros((len(train), 13))
	test_feature = np.zeros((len(test), 13))

	for i in range(len(train)):
		sample_rate, X = scipy.io.wavfile.read(train[i])
		ceps, mspec, spec = mfcc(X, fs=sample_rate)
		num_ceps = len(ceps)
		mfcc_coeff = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
		train_feature[i] = mfcc_coeff

		# plot
		plt.figure(1)
		if train[i].find("chinese"):
			plt.plot(range(13), mfcc_coeff, 'r')
		elif train[i].find("english"):
			plt.plot(range(13), mfcc_coeff, 'b')

	#plt.show()

	for i in range(len(test)):
		sample_rate, X = scipy.io.wavfile.read(test[i])
		ceps, mspec, spec = mfcc(X, fs=sample_rate)
		num_ceps = len(ceps)
		mfcc_coeff = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
		test_feature[i] = mfcc_coeff

	return train_feature, test_feature


if __name__ == "__main__":
	train = np.array(["./Data/train/chinese1.wav","./Data/train/chinese2.wav","./Data/train/english1.wav","./Data/train/english2.wav"])
	test = np.array(["./Data/test/chinese3.wav","./Data/test/chinese4.wav","./Data/test/english3.wav","./Data/test/english4.wav"])
	[trainF, testF] = ExtreactMFCC(train, test)

	trainFileName = "./Feature/train.dev"
	testFileName  = "./Feature/test.dev"
	Written2File(trainFileName, trainF)
	Written2File(testFileName, testF)

