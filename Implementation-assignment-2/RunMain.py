import numpy as np
import matplotlib.pyplot as plt

import NaiveBayes as nb

def RunMain(val):
	print 'Yor are in RunMain\n'

	nbModel = nb.NAIVE_BAYES_MODEL()
	nbModel.estimatePy_MLE()

	

if __name__ == "__main__":
	RunMain(3)