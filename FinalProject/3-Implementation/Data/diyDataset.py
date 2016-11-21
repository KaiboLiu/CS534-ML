'''
This code is made by KaiboLiu
to automatically creat training and dev dataset with audio files from original dataset
for final project of CS534
'''

#!/usr/bin/env python
# coding:utf-8
import numpy as np
import os
import shutil 
import random
import time


def LoadData(dataDir,lang):
    # read input data for training & testing
    
    data_path = dataDir+'list_'+lang+'.data'
    #wav_path  = dataDir+lang 
    
    Data = []
    f = open(data_path)
    for line in f.readlines():  
        Data.append(line.strip().split('\t'))
    Data = np.array(Data)
    f.close()
    
    return Data

def WriteFile(fileName, Data):
	f = open(fileName,"w")
	for line in Data:
		f.write(line[0]+'\t'+line[1]+'\n')
		
	f.close()

def RunCreat():

	time.clock()
	t00 = float(time.clock())
	
	srcDir = './wav/'
	dstDir = './diyDataset/'
	lang_list= ['cmn','eng','engStory','deu','fra','jpn','rus'] 
	#cmn  Chinese(Mandarin),female:500
	#eng  English,male:500
	#engStory:a story vedio splitted in to 688 wavs. 
	#deu  German,male:400
	#fra  French,male:400
	#jpn  Japanese,female:400
	#rus  Russian,male:400


	if os.path.exists(dstDir):
		shutil.rmtree(dstDir)  	#delete the dir of path and all files in it
	os.makedirs(dstDir)		#creat the dir for new diy dataset
	os.makedirs(dstDir+'train/')
	os.makedirs(dstDir+'dev/')

	'''use following two lists to decide the size of new dataset. 
	If you set some of the element >0, then the corresponding language is added to dataset'''
	# origin data samples is [500,500,688,400,400,400,400], can be divided into training and dev in each language
	n_list_train = 	[300,300,388,0,0,0,0]#[3,3,3,2,4,0,3]
	n_list_dev   = 	[200,200,300,0,0,0,0]#[1,1,1,1,1,0,1]

	dataTrain = []
	dataDev = []

	n_train = sum(n_list_train)
	n_dev = sum(n_list_dev)

	for i in range(len(lang_list)):
		t0 = float(time.clock())
		n_sample = n_list_dev[i]+n_list_train[i]
		if n_sample == 0:
			continue
		if n_list_train[i] == 0:
			continue
		Data = LoadData(srcDir,lang_list[i])

		if n_sample > len(Data):
			newData = dataList
		else:
			newData = np.array(random.sample(Data,n_sample))   #randomly choose n_sample data to newData

		# start to build training data with language[i]
		for j in range(n_list_train[i]):
			index, lang = newData[j,0], newData[j,1]
			if index[0] == 'S':
				lang = 'engStory'
			dataTrain.append(newData[j].tolist())  #add this speech information into diy training data(list)
			shutil.copy(srcDir+lang+'/'+index+'.wav', dstDir+'train/')

		# start to build dev data with language[i]
		for j in range(n_list_train[i],n_sample):
			index, lang = newData[j,0], newData[j,1]
			if index[0] == 'S':
				lang = 'engStory'
			dataDev.append(newData[j].tolist())  #add this speech information into diy dev data(list)
			shutil.copy(srcDir+lang+'/'+index+'.wav', dstDir+'dev/')

		t1 = float(time.clock())
		print 'from %s, copy %d data into train, and %d data into dev, using %.4fs.' %(lang_list[i],n_list_train[i],n_list_dev[i],t1-t0)
		
	#print dataTrain
	#print dataDev

	WriteFile(dstDir+'speech_list.train', random.sample(dataTrain,n_train))
	WriteFile(dstDir+'speech_list.dev', random.sample(dataDev,n_dev))

	t1 = float(time.clock())
	print '[done] total runtime is %.4fs. \n' % (t1-t00)
		
		

if __name__ == "__main__":
	RunCreat()
