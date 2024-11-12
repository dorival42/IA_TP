import numpy as np
from PIL.Image import *
import os
from os.path import join

# importation of all the functions in func.py
from func import *

if __name__ == '__main__':
    
    nbIter = 4  # number of iterations for EM
    K      = 2  # number of classes

    reposource = './sources'
    reporesult = './results'
    
    #imageorigname    = join(reposource, 'lapin.png')
    imageorigname    = join(reposource, 'cible_128.png')
    basename         = os.path.basename(imageorigname)
    filename, file_extension = os.path.splitext(basename)
    imageNoisyname   = join(reporesult, filename+'_bruit'      +file_extension)
    histoname        = join(reporesult, filename+'_bruit_histo'+file_extension)
    imageclassifname = join(reporesult, filename+'_classif'    +file_extension)
    
    # Parameters of MM: mean, variance and a priori proba
    meanTabIter = np.zeros(shape=(nbIter, K))
    varTabIter  = np.ones(shape=(nbIter, K))
    piTabIter   = np.ones(shape=(nbIter, K)) / K
    
    # Error rate according to EM iterations
    classErrorRateTab  = np.zeros(shape=(nbIter, K))
    globalErrorRateTab = np.zeros(shape=(nbIter))
    
    # Likelihood and auxiliary Q computing
    likelihood = np.zeros(shape=(nbIter))
    auxiliaryQ = np.zeros(shape=(nbIter))
    
    # Reading the noisy target image and converting it to a numpy array
    imageNoisy = np.array(open(imageNoisyname), dtype=float)
    [L, C] = np.shape(imageNoisy)
    
    # Reading the two-classes target image and converting it to a numpy array
    # This image is only required for comparison purpose with the segmented image
    imageBW = np.array(open(imageorigname), dtype=float)
    
    ##########################################################################
    # Initialization of parameters
    iteration = 0
    meanTabIter[iteration, :], varTabIter[iteration, :], piTabIter[iteration, :] = InitParam(K, imageNoisy)
    
    # Mapping the class image between 0...K-1 to 0...255 (to see the difference between the classes when show as an image)
    imageNoisy_classif = BayesianClassif(imageNoisy, meanTabIter[iteration, :], varTabIter[iteration, :], piTabIter[iteration, :])
    
    # Saving the segmented image + the estimated mixture at this stage
    if iteration % 10 == 0 or iteration == nbIter-1:
        Savings(join(reporesult, basename), iteration, imageNoisy, imageNoisy_classif, meanTabIter[iteration, :], varTabIter[iteration, :], piTabIter[iteration, :])

    # Error rate computing
    globalErrorRateTab[iteration], classErrorRateTab[iteration, :] = errorRateComputing(imageBW, imageNoisy_classif, K)
    
    # likelihood and auxiliary Q computing
    likelihood[iteration], auxiliaryQ[iteration] = likelihood_Q_computing(imageNoisy, meanTabIter[iteration, :], varTabIter[iteration, :], piTabIter[iteration, :])
    

    ##########################################################################
    # EM iterations
    for iteration in range(1, nbIter):
        print('--->iteration=', iteration)
        
        EM_Iter(iteration, imageNoisy, meanTabIter, varTabIter, piTabIter)

        # Mapping the class image between 0...K-1 to 0...255 (to see the difference between the classes when show as an image)
        imageNoisy_classif = BayesianClassif(imageNoisy, meanTabIter[iteration, :], varTabIter[iteration, :], piTabIter[iteration, :])
        
        # Saving the segmented image plus the estimated mixture
        if iteration % 10 == 0 or iteration == nbIter-1:
            Savings(join(reporesult, basename), iteration, imageNoisy, imageNoisy_classif, meanTabIter[iteration, :], varTabIter[iteration, :], piTabIter[iteration, :])

        # Error rate computing
        globalErrorRateTab[iteration], classErrorRateTab[iteration, :] = errorRateComputing(imageBW, imageNoisy_classif, K)
        
        # likelihood and auxiliary Q computing
        likelihood[iteration], auxiliaryQ[iteration] = likelihood_Q_computing(imageNoisy, meanTabIter[iteration, :], varTabIter[iteration, :], piTabIter[iteration, :])


    # Drawing curves: evolution of parameters. Don't forget likelihood
    pathToSave = join(reporesult, filename)
    DrawCurvesParam(nbIter, K, pathToSave, meanTabIter, varTabIter, piTabIter)
    DrawCurvesError(nbIter, K, pathToSave, classErrorRateTab, globalErrorRateTab)
    DrawLQ(nbIter, K, pathToSave, likelihood, auxiliaryQ)

    # Saving the segmented image plus the estimated mixture
    Savings(join(reporesult, basename), iteration, imageNoisy, imageNoisy_classif, meanTabIter[iteration, :], varTabIter[iteration, :], piTabIter[iteration, :])
    
    print('Mean error rate by class =', classErrorRateTab [nbIter-1, :])
    print('Global mean error rate   =', globalErrorRateTab[nbIter-1])
    
    print('Pi estimated', piTabIter[iteration, :])
    print('Mean estimated', meanTabIter[iteration, :])
    print('Std estimated', np.sqrt(varTabIter[iteration, :]))
    