#This code is made by KaiboLiu, to automatically download audio files from tatoeba.org

# coding:utf-8
import re
import numpy as np
import os
import urllib2,urllib  

def AudioCrawl(url_0,page1,page2,lang,saveDir,subDir):
			
	if not os.path.exists(saveDir+lang+'_'+subDir):
		os.makedirs(saveDir+lang+'_'+subDir)	
	f = open(saveDir+lang+'_list.'+subDir[:-1],"w")
	#f = open(saveDir+list.'+subDir[:-1],"a")

	for i in range(page1,page2+1):
		if i > 1:
			url = url_0 + '/page:' + str(i) 
		else:
			url = url_0
		req = urllib2.Request(url)  
		content = urllib2.urlopen(req).read() 
		match1 = re.compile(r'(?<=href=["]).*?(?=["])')  #<a href="https://audio.tatoeba.org/sentences/cmn/332420.mp3"
		rawlv1 = re.findall(match1,content)
		
		mp3_url = []
		index = []
		for line in rawlv1:
			if line[-3:] == 'mp3':	
				mp3_url.append(line)
				index.append(line[line.rfind('/')+1:])
				#print line
		print url, len(mp3_url)
		'''
		#add translations to mp3_url
		for line in rawlv2:
			mp3_url.append(line)
		print len(mp3_url)
		mp3_url = np.array(mp3_url)
		mp3_url = np.transpose(np.reshape(mp3_url,(2,-1)))
		'''	
		for i in range(len(index)):
			f.write(index[i][:-4] + '\t'+ lang+'\n')
			#filepath = saveDir +subDir + index[i]
			filepath = saveDir +lang+'_'+subDir + index[i]			
			urllib.urlretrieve(mp3_url[i], filepath)	

	f.close()

def RunBatch():
	url_cmn = 'http://tatoeba.org/eng/sentences/with_audio/cmn'  # or 'http://tatoeba.org/eng/sentences/with_audio/cmn/page:2', 34 pages in total
	url_eng = 'http://tatoeba.org/eng/sentences/with_audio/eng'  # or 'http://tatoeba.org/eng/sentences/with_audio/eng/page:2', 4939 pages in total
	page1, page2 = 1, 6
	'''
	lang = 'cmn'
	AudioCrawl(url_cmn,page1,page2,lang,'./autosave/','train/')
	AudioCrawl(url_cmn,page2+10,page2+11,lang,'./autosave/','test/')
	'''
	lang = 'eng'
	AudioCrawl(url_eng,page1,page2,lang,'./autosave_mp3/','train/')
	AudioCrawl(url_eng,page2+10,page2+11,lang,'./autosave_mp3/','test/')


if __name__ == "__main__":
	RunBatch()
