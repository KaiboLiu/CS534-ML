#!/usr/bin/env python
import numpy as np
import sys
import operator

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



