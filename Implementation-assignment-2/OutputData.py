#!/usr/bin/env python
import numpy as np
import sys
import operator
import matplotlib.pyplot as plt

import pylab

'''
# for machine learning assignment2
# save a 0/1 vector to a .dev file
# mapping 0/1 to str0/str1, and save each work in a line
'''
def WritenFile_dev(fileName, vecV, str0, str1):
	f = open(fileName,"w")
	for v in vecV:
		if(v == 0):
			line = str0
			f.write(line + "\n")
		else:
			line = str1
			f.write(line + "\n")
	f.close()

def get_colour(color):
    '''
    b---blue   c---cyan  g---green    k----black
    m---magenta r---red  w---white    y----yellow
'''
    color_set = ['r--','b--','m--','g--','c--','k--','y--']
    return color_set[color % 7]

def Save2Figure_semilogs(fileName, figNum, xVec, yVec, labelName, labelLim, useLog, yChgName = []):
	yNum = len(yVec)
	plt.figure(figNum)
	if yNum == 1:
		if useLog == 1:
			plt.semilogx(xVec, yVec[0], get_colour(0))
		else:
			plt.plot(xVec, yVec[0], get_colour(0))
		#ax.semilogs(xVec, yVec[0])
	else:
		for i in range(yNum):
			if useLog == 1:
				plt.semilogs(xVec, yVec[i], get_colour(i),label=yChgName[i])
			else:
				plt.plot(xVec, yVec[i], get_colour(i), label=yChgName[i])
			color += 1
	plt.xlabel(labelName[0])
	plt.ylabel(labelName[1]) 
	plt.xlim(labelLim[0], labelLim[1])# set axis limits
	plt.ylim(labelLim[2], labelLim[3])

	#plt.legend(loc='upper right')
	plt.grid(True)

	plt.savefig(fileName)
    

