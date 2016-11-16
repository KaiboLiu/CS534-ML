import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt
import librosa

def Written2File(fileName, vecV):
	f = open(fileName, "w")
	for v in vecV:
		line = " ".join([str(fea) for fea in v.tolist()])
		f.write(line + "\n")
	f.close()

def ExtreactMFCC(train, test):
	feature_dim = 13
	train_feature = np.zeros((len(train), feature_dim+1))
	test_feature = np.zeros((len(test), feature_dim+1))

	for i in range(len(train)):
		sample_rate, X = scipy.io.wavfile.read(train[i])
		ceps, mspec, spec = mfcc(X, fs=sample_rate)
		num_ceps = len(ceps)
		mfcc_coeff = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
		if train[i].find("chinese") > 0:
			mfcc_coeff_withClass = np.append(mfcc_coeff, 0)
		elif train[i].find("english") > 0:
			mfcc_coeff_withClass = np.append(mfcc_coeff, 1)
		train_feature[i] = mfcc_coeff_withClass

		'''
		# plot
		plt.figure(1)
		if train[i].find("chinese"):
			plt.plot(range(feature_dim), mfcc_coeff, 'r')
		elif train[i].find("english"):
			plt.plot(range(feature_dim), mfcc_coeff, 'b')
		'''

	#plt.show()

	for i in range(len(test)):
		sample_rate, X = scipy.io.wavfile.read(test[i])
		ceps, mspec, spec = mfcc(X, fs=sample_rate)
		num_ceps = len(ceps)
		mfcc_coeff = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
		if test[i].find("chinese") > 0:
			mfcc_coeff_withClass = np.append(mfcc_coeff, 0)
		elif test[i].find("english") > 0:
			mfcc_coeff_withClass = np.append(mfcc_coeff, 1)
		test_feature[i] = mfcc_coeff_withClass

	return train_feature, test_feature

def ExtractFeaturesByLibrosa(train, test):
	for i in range(len(train)):
		print "\nfilename:", train[i]
		y, sr = librosa.load(train[i])

		tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
		print "tempo:", tempo

		beat_times = librosa.frames_to_time(beat_frames, sr=sr)
		print "beat_times:", beat_times.shape

		chromagram = librosa.feature.chroma_stft(y, sr=sr)
		print "chromagram:", chromagram.shape

		Qchromagram = librosa.feature.chroma_cqt(y, sr=sr)
		print "Qchromagram:", Qchromagram.shape

		chromagram_cens = librosa.feature.chroma_cens(y, sr=sr)
		print "chromagram_cens:", chromagram_cens.shape

		melspec = librosa.feature.melspectrogram(y, sr=sr)
		print "melspectrogram:", melspec.shape

		mfcc = librosa.feature.mfcc(y, sr=sr)
		print "mfcc:", mfcc.shape

		rmse = librosa.feature.rmse(y)
		print "rmse:", rmse.shape

		centroid = librosa.feature.spectral_centroid(y, sr=sr)
		print "spectral_centroid:", centroid.shape

		bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr)
		print "spectral_bandwidth:", bandwidth.shape

		contrast = librosa.feature.spectral_contrast(y, sr=sr)
		print "spectral_contrast:", contrast.shape

		rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
		print "rolloff:", rolloff.shape

		poly_features = librosa.feature.poly_features(y, sr=sr)
		print "poly_features:", poly_features.shape

		tonnetz = librosa.feature.tonnetz(y, sr=sr)
		print "tonnetz:", tonnetz.shape

		zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
		print "zero_crossing_rate:", zero_crossing_rate.shape


if __name__ == "__main__":
	train = np.array(["./Data/train/chinese1.wav","./Data/train/chinese2.wav","./Data/train/english1.wav","./Data/train/english2.wav"])
	test = np.array(["./Data/test/chinese3.wav","./Data/test/chinese4.wav","./Data/test/english3.wav","./Data/test/english4.wav"])
	#[trainF, testF] = ExtreactMFCC(train, test)
	ExtractFeaturesByLibrosa(train, test)

	trainFileName = "./Feature/train.dev"
	testFileName  = "./Feature/test.dev"
	#Written2File(trainFileName, trainF)
	#Written2File(testFileName, testF)