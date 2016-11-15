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

def ExtractTempoBeat(train, test):
	for i in range(len(train)):
		print "\nfilename:", train[i]
		y, sr = librosa.load(train[i])
		#print "waveform:", y # waveform
		#print "sampling rate", sr # sampling rate

		tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
		print "tempo:", tempo
		#print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

		beat_times = librosa.frames_to_time(beat_frames, sr=sr)
		#print "beat_frames:", beat_frames
		print "beat_times:", np.average(beat_times)
		#print('Saving output to beat_times.csv')
		#librosa.output.times_csv('beat_times.csv', beat_times)


if __name__ == "__main__":
	train = np.array(["./Data/train/chinese1.wav","./Data/train/chinese2.wav","./Data/train/english1.wav","./Data/train/english2.wav"])
	test = np.array(["./Data/test/chinese3.wav","./Data/test/chinese4.wav","./Data/test/english3.wav","./Data/test/english4.wav"])
	[trainF, testF] = ExtreactMFCC(train, test)
	ExtractTempoBeat(train, test)

	trainFileName = "./Feature/train.dev"
	testFileName  = "./Feature/test.dev"
	#Written2File(trainFileName, trainF)
	#Written2File(testFileName, testF)