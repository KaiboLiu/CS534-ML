import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt

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

'''
if __name__ == "__main__":
	train = np.array(["chinese1.wav","chinese2.wav","english1.wav","english2.wav"])
	test = np.array(["chinese3.wav","chinese4.wav","english3.wav","english4.wav"])
	ExtreactMFCC(train, test)
'''