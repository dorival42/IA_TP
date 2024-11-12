import numpy as np
from PIL.Image import *
from scipy.stats import norm
import matplotlib.pyplot as plt


fontS  = 12     # font size
colors = ['r', 'g', 'b', 'y', 'm']

def InitParam(K, imageNoisy):

	meanTab = np.zeros(shape=(K))
	varTab  = np.zeros(shape=(K))
	piTab   = np.zeros(shape=(K))

	#################################
	print("InitParam / TO DO")
	#################################
	
	return meanTab, varTab, piTab


def BayesianClassif(imageNoisy, meanTab, varTab, piTab):
	
	# Sorting all the values according to the meanvalues
	isort   = np.argsort(meanTab) # isort contains the re-indexing for sorting
	meanTab = meanTab[isort]      # applying the re-indexing
	varTab  = varTab[isort]
	piTab   = piTab[isort]

	L, C = np.shape(imageNoisy)
	K    = np.size(meanTab)

	discrimTab       = np.zeros(shape=(K))
	BayesianThresold = np.zeros(shape=(256))
	
	for ndg in range(256):
		for k in range(K):
			discrimTab[k] = piTab[k] * norm.pdf(ndg, loc=meanTab[k], scale=np.sqrt(varTab[k]))
		discrimTab /= np.sum(discrimTab)
		BayesianThresold[ndg] = np.argmax(discrimTab)*255
	#print(BayesianThresold)

	# Mapping
	imageNoisy_classif = np.zeros(shape=(L, C))
	for l in range(L):
		for c in range(C):
			imageNoisy_classif[l, c] = BayesianThresold[imageNoisy[l, c].astype(int)]
		
	return imageNoisy_classif



def EM_Iter(iteration, imageNoisy, meanTabIter, varTabIter, piTabIter):

	L, C = np.shape(imageNoisy)
	K    = np.size(meanTabIter[0, :])
	N = L*C

	#################################
    print("EM_Iter / TO DO")
    #################################


def likelihood_Q_computing(imageNoisy, meanTab, varTab, piTab) :

	L, C = np.shape(imageNoisy)
	K    = np.size(meanTab)

	Likelihood = 0.
	AuxiliaryQ = 0.

	#################################
    print("likelihood_Q_computing / TO DO")
    #################################
					
	return Likelihood, AuxiliaryQ


def errorRateComputing(imageBW, imageNoisy_classif, K):

	L, C = np.shape(imageBW)
	
	cptRateTab = np.zeros(shape=(K))
	eRateTab   = np.zeros(shape=(K))
	eRate      = 0
	for l in range(L):
		for c in range(C):
			if imageBW[l,c] != imageNoisy_classif[l,c]:
				cptRateTab[ (imageBW[l,c]/255).astype(int) ] += 1
				eRate += 1

	A0   = np.size(np.where(imageBW == 0)[0])
	eRateTab[0] = cptRateTab[0]/A0
	eRateTab[1] = cptRateTab[1]/(L*C-A0)
	eRate    /= L*C

	return eRate, eRateTab


def Savings(basename, iteration, imageNoisy, imageNoisy_classif, meanTab, varTab, piTab):

	L, C = np.shape(imageNoisy)
	K    = np.size(meanTab)

	# Saving the segmented image    
	imagebruitname = basename + '_bruit_classif_' + str(K) + '_iter_' + str(iteration)+ '.png'
	imageNoisy_Classif = fromarray(np.uint8(imageNoisy_classif))
	imageNoisy_Classif.save(imagebruitname)
	
	# Getting the min and max pixel value
	min_im = int(np.min(np.min(imageNoisy)))
	max_im = int(np.max(np.max(imageNoisy)))
	
	# Histogram
	imageNoisy = fromarray(np.uint8(imageNoisy))
	imhisto = imageNoisy.histogram()
	imhistoNorm = [x / (L*C) for x in imhisto]
	fig, ax = plt.subplots(1, 1)
	ax.bar(x=range(min_im, max_im), height=imhistoNorm[min_im:max_im], label='Normalized im histo')
	
	# Mixture of estimated Gaussians
	x = np.linspace(min_im, max_im, 200)
	CondPdfTab = np.zeros(shape=(len(x)))
	for k in range(K):
		CondPdfTab += piTab[k] * norm.pdf(x, loc=meanTab[k], scale=np.sqrt(varTab[k]))
	ax.plot(x, CondPdfTab, 'k', lw=3, alpha=0.9, label='Mixture at iter ' + str(iteration))
	ax.legend()
	histobruitname = basename + '_histo_melange_' + str(K) + '_iter_' + str(iteration)+ '.png'
	fig.savefig(histobruitname)
	plt.close()


def DrawCurvesParam(nbIter, K, pathToSave, meanTabIter, varTabIter, piTabIter):

	fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
	for k in range(K):
		ax1.plot(range(nbIter), meanTabIter[:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
		ax2.plot(range(nbIter), varTabIter [:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
		ax3.plot(range(nbIter), piTabIter  [:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))

		ax1.set_ylabel('mu',       fontsize=fontS)
		ax2.set_ylabel('sigma**2', fontsize=fontS)
		ax3.set_ylabel('pi',       fontsize=fontS)
		ax1.legend()

	# figure saving
	plt.xlabel('EM iterations', fontsize=fontS)
	plt.savefig(pathToSave + '_EvolParam.png', bbox_inches='tight', dpi=150)
	plt.close()
	
	
def DrawCurvesError(nbIter, K, pathToSave, classErrorRateTab, globalErrorRateTab):
	fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
	for k in range(K):
		ax1.plot(range(nbIter), classErrorRateTab[:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
		ax2.plot(range(nbIter), globalErrorRateTab, lw=1, alpha=0.9, color='k', label='global')

		ax1.set_ylabel('% error', fontsize=fontS)
		ax2.set_ylabel('% error', fontsize=fontS)
		ax1.legend()

	# figure saving
	plt.xlabel('EM iterations', fontsize=fontS)
	plt.savefig(pathToSave + '_EvolError.png', bbox_inches='tight', dpi=150)
	plt.close()
	
	
def DrawLQ(nbIter, K, pathToSave, likelihood, auxiliaryQ):

	fig, ax1 = plt.subplots(nrows=1, ncols=1)

	ax1.plot(range(nbIter), auxiliaryQ, lw=1, alpha=0.9, color='g', label='Auxiliary Q')
	ax1.plot(range(nbIter), likelihood, lw=1, alpha=0.9, color='r', label='Log likelihood')
	ax1.legend()

	# figure saving
	plt.xlabel('EM iterations', fontsize=fontS)
	plt.savefig(pathToSave + '_LQ.png', bbox_inches='tight', dpi=150)
	plt.close()
