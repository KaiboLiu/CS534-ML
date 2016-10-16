import numpy as np

def removed_word(wordNum, idx_list):
	labelVec = np.ones(wordNum)
	if type(idx_list[0]) is list:
		for k in range(len(idx_list)):
			labelVec[idx_list[k]] = 0
	elif type(idx_list[0]) is int:
		labelVec[idx_list] = 0
	removedIdx = np.where(labelVec == 0)
	selectedIdx = np.where(labelVec > 0)
	redFeaNum = len(selectedIdx[0])
	return removedIdx, labelVec, redFeaNum

def find_top_words(classNum, wordNum, rank, Pwy):
	removedIdx = []
	for k in range(classNum):
		max_idx = Pwy[k].argsort()[-rank:][::-1]
		removedIdx.append(max_idx.tolist())
	[removedIdx, labelVec, redFeaNum] =  removed_word(wordNum, removedIdx)
	return removedIdx, labelVec, redFeaNum

def find_std_zero_words(wordNum, threshold, Pwy):
	Pwy_matrix = np.array(Pwy)
	Pwy_std = Pwy_matrix.std(0)
	print "std avg.",  np.average(Pwy_std)
	removed = np.where(Pwy_std == threshold)
	removedIdx = removed[0].tolist()
	[removedIdx, labelVec, redFeaNum] =  removed_word(wordNum, removedIdx)
	return removedIdx, labelVec, redFeaNum

def find_low_tfidf_words(wordNum, threshold, Pwy_b, Pwy_m):
	TF = np.sum([Pwy_m[0], Pwy_m[1]], axis=0)
	#plt.plot(TF)
	#plt.show()
	DF = np.amin([Pwy_b[0], Pwy_b[1]], axis=0)
	#plt.plot(DF)
	#plt.show()
	TFIDF = np.divide(TF,abs(DF+1))
	tfidf_list = TFIDF.tolist()
	removedIdx = ([ n for n,i in enumerate(tfidf_list) if i < threshold])
	[removedIdx, labelVec, redFeaNum] =  removed_word(wordNum, removedIdx)
	return removedIdx, labelVec, redFeaNum