import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt
import librosa
import pandas

def LoadData():
    # read input data for training & testing
    dataDir    = "./Data/"
    trainFile  = "train.txt"
    testFile   = "test.txt"
    train = pandas.read_csv(dataDir+trainFile, sep=",", names=["filename","label"])
    test = pandas.read_csv(dataDir+testFile, sep=",", names=["filename","label"])

    return [train.filename, train.label, test.filename, test.label]

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

def ExtractFeaturesByLibrosa(sample, label, filePath):
	sampleF = []

	for i in range(len(sample)):
		print i
		feature = []
		filename = filePath + sample[i].astype('str') + ".wav"
		y, sr = librosa.load(filename)
		'''
		tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
		print "tempo:", tempo

		beat_times = librosa.frames_to_time(beat_frames, sr=sr)
		print "beat_times:", beat_times.shape
		'''
		chromagram = librosa.feature.chroma_stft(y, sr=sr)
		#print "chromagram:", np.average(chromagram, 1)
		feature.extend(np.average(chromagram, 1))

		Qchromagram = librosa.feature.chroma_cqt(y, sr=sr)
		#print "Qchromagram:", np.average(Qchromagram, 1)
		feature.extend(np.average(Qchromagram, 1))

		chromagram_cens = librosa.feature.chroma_cens(y, sr=sr)
		#print "chromagram_cens:", chromagram_cens.shape
		feature.extend(np.average(chromagram_cens, 1))

		melspec = librosa.feature.melspectrogram(y, sr=sr)
		#print "melspectrogram:", melspec.shape
		feature.extend(np.average(melspec, 1))

		mfcc = librosa.feature.mfcc(y, sr=sr)
		#print "mfcc:", mfcc.shape
		feature.extend(np.average(mfcc, 1))

		rmse = librosa.feature.rmse(y)
		#print "rmse:", rmse.shape
		feature.extend(np.average(rmse, 1))

		centroid = librosa.feature.spectral_centroid(y, sr=sr)
		#print "spectral_centroid:", centroid.shape
		feature.extend(np.average(centroid, 1))

		bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr)
		#print "spectral_bandwidth:", bandwidth.shape
		feature.extend(np.average(bandwidth, 1))

		contrast = librosa.feature.spectral_contrast(y, sr=sr)
		#print "spectral_contrast:", contrast.shape
		feature.extend(np.average(contrast, 1))

		rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
		#print "rolloff:", rolloff.shape
		feature.extend(np.average(rolloff, 1))

		poly_features = librosa.feature.poly_features(y, sr=sr)
		#print "poly_features:", poly_features.shape
		feature.extend(np.average(poly_features, 1))

		tonnetz = librosa.feature.tonnetz(y, sr=sr)
		#print "tonnetz:", tonnetz.shape
		feature.extend(np.average(tonnetz, 1))

		zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
		#print "zero_crossing_rate:", zero_crossing_rate.shape
		feature.extend(np.average(zero_crossing_rate, 1))

		if label[i].find("cmn") == 0:
			feature.append(0)
		elif label[i].find("eng") == 0:
			feature.append(1)

		sampleF.append(feature)

	return np.array(sampleF)

if __name__ == "__main__":
	#[trainF, testF] = ExtreactMFCC(train, test)
	[trainX, trainY, testX, testY] = LoadData()

	fileDir    = "./Data/"
	trainFolder  = "train/"
	testFolder   = "test/"
	trainF = ExtractFeaturesByLibrosa(trainX, trainY, fileDir+trainFolder)
	testF = ExtractFeaturesByLibrosa(testX, testY, fileDir+testFolder)

	trainFileName = "./Feature/train.dev"
	testFileName  = "./Feature/test.dev"
	Written2File(trainFileName, trainF)
	Written2File(testFileName, testF)