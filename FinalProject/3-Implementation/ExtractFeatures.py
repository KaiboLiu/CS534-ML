import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt
import librosa
import random
import pandas

def LoadData():
    # read input data for training & testing
    dataDir    = "./Data/diyDataset/"
    trainFile  = "speech_list.train"
    testFile   = "speech_list.test"
    train = pandas.read_csv(dataDir+trainFile, sep="\t", names=["filename","label"])
    test = pandas.read_csv(dataDir+testFile, sep="\t", names=["filename","label"])

    return [train.filename, train.label, test.filename, test.label]

def Written2File(fileName, vecV):
	f = open(fileName, "w")
	for v in vecV:
		line = " ".join([str(fea) for fea in v.tolist()])
		f.write(line + "\n")
	f.close()

def ExtractFeaturesByLibrosa(sample, label, filePath):
	sampleF = []

	for i in range(len(sample)):
		print i
		feature = []
		filename = filePath + sample[i] + ".wav"
		y, sr = librosa.load(filename)

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
		else:
			print "Test file"

		sampleF.append(feature)

	return np.array(sampleF)

def RunMain():
	'''
	[trainX, trainY, testX, testY] = LoadData()

	trainFolder  = "./Data/diyDataset/train/"
	testFolder   = "./Data/diyDataset/test/"
	trainF = ExtractFeaturesByLibrosa(trainX, trainY, trainFolder)
	testF = ExtractFeaturesByLibrosa(testX, testY, testFolder)

	trainFeature = "./Feature/train.dev"
	testFeature  = "./Feature/test.dev"
	Written2File(trainFeature, trainF)
	Written2File(testFeature, testF)
	'''

	[trainX, trainY, testX, testY] = LoadData()

	trainFolder  = "./Data/diyDataset/train/"
	testFolder   = "./Data/diyDataset/test/"
	trainF = ExtractFeaturesByLibrosa(trainX, trainY, trainFolder)
	testF = ExtractFeaturesByLibrosa(testX, testY, testFolder)

	trainLen = len(trainF)
	rdmIdx1  = random.sample(range(trainLen), trainLen)
	new_trainF = trainF[rdmIdx1[:]]

	testLen = len(testF)
	rdmIdx2 = random.sample(range(testLen), testLen)
	new_testF = testF[rdmIdx2[:]]

	trainFeature = "./Feature/train.dev"
	testFeature  = "./Feature/test.dev"
	Written2File(trainFeature, new_trainF)
	Written2File(testFeature, new_testF)

if __name__ == "__main__":
	RunMain()
