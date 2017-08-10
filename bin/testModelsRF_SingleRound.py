#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:34:08 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')

import parcellearning.classifier_utilities as cu
import parcellearning.loaded as ld
import parcellearning.malp as malp
import parcellearning.MixtureModel as MM

from keras.models import load_model

import nibabel as nb
import numpy as np

import glob
import os
import pickle

from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()


def pickleLoad(inFile):
    
    with open(inFile,'r') as inputFile:
        data = pickle.load(inputFile)
    
    return data

def loadTest(model,yObject,yMatch):
        
        """
        Method to load the test data into the object.  We might be interested
        in loading new test data into, so we have explicitly defined this is
        as a method.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain      
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
        """
        
        nf = []
        for f in model.features:
            if f != 'label':
                nf.append(f)
                
        print 'Model features: {}'.format(model.features)
        print 'Load test features: {}'.format(nf)

        # load test subject data, save as attribtues
        tObject = ld.loadH5(yObject,*['full'])
        ID = tObject.attrs['ID']

        parsedData = ld.parseH5(tObject,nf)
        tObject.close()

        data = parsedData[ID]
        mtd = cu.mergeFeatures(data,nf)

        threshed = ld.loadMat(yMatch)

        ltvm = cu.vertexMemberships(threshed,180)

        return [threshed,mtd,ltvm]
    
    
"""

parallelPredictRF and atlasPredictRF rely on the internal predict method of
the MultiAtlas and Atlas objects.  In the event that we have serialization
problems (after saving the models, changing the code for MultiAtlas and Atlas)
we need different predict methods that are independent of the state at the 
point of serialization.
"""
def parallelPredictRF(models,yTestObject,yTestMatch,predictCase = 0):
    
    if predictCase:
    
        predictedLabels = Parallel(n_jobs=NUM_CORES)(delayed(atlasPredictRF)(models[i],
                                   yTestObject,yTestMatch,'FORESTS') for i,m in enumerate(models))
    else:
        predictedLabels = Parallel(n_jobs=NUM_CORES)(delayed(atlasPredictBaseCase)(models[i],
                                   yTestObject,yTestMatch,'FORESTS') for i,m in enumerate(models))

    predictedLabels = np.column_stack(predictedLabels)
    classification = []
        
    for i in np.arange(predictedLabels.shape[0]):
        
        L = list(predictedLabels[i,:])
        maxProb = max(set(L),key=L.count)
        classification.append(maxProb)
    
    classification = np.asarray(classification)

    return classification

def atlasPredictRF(mod,yObject,yMatch,sfmt):
    
    """
    Method to predict labels of test data using model predict method.
    
    Parameters:
    - - - - -
        mod : current random forest model
        
        yObject : testing subject training object
        
        yMatch : test subject matching matrix
        
        mm : matching matrix (binary or frequency)
    """
    
    [mm,mtd,ltvm] = loadTest(mod,yObject,yMatch)
    
    mod.predict(mtd,mm,ltvm,softmax_type = sfmt)
    
    P = mod.predicted
    
    return P

def atlasPredictBaseCase(mod,yObject,yMatch,sfmt,**kwargs):
    
    """
    Method to predict labels of test data using external method.
    
    Parameters:
    - - - - -
        mod : current random forest model
        
        yObject : testing subject training object
        
        yMatch : test subject matching matrix
        
        mm : matching matrix (binary or frequency)
    """
    
    [mm,mtd,ltvm] = loadTest(mod,yObject,yMatch)
    softmax_type = sfmt

    if kwargs:
        if 'power' in kwargs.keys():
            p = kwargs['power']
        else:
            p = 1
    else:
        p = 1

    funcs = {'BASE': malp.baseSoftMax,
             'TREES': malp.treeSoftMax,
             'FORESTS': malp.forestSoftMax}
    
    labels = mod.labels
    neighbors = mod.neighbors
    
    R = 180
    [xTest,yTest] = mtd.shape
    if yTest != mod.input_dim:
        raise Warning('Test data does not have the same number \
                      features as the training data.')

    # initialize prediction dictionary
    baseline = np.zeros((mtd.shape[0],R+1))

    mm = np.power(mm,p)

    for lab in labels:
        if lab in neighbors.keys():
            
            members = ltvm[lab]
            memberData = mtd[members,:]
            estimator = mod.models[lab]
            
            if len(members) > 0:
                preds = funcs[softmax_type](estimator,members,memberData,mm,R)
                baseline = cu.updatePredictions(baseline,members,preds)
            
    predicted = np.argmax(baseline,axis=1)
    
    return predicted

parser = argparse.ArgumentParser(description='Compute random forest predictions.')
# Parameters for input data
parser.add_argument('-r','--round',help='Group of subjects to process.',type=int,required=True)
parser.add_argument('-f','--frequencyBased',help='Whether to use frequency-based neighborhood constraint.',
                    default=False,type=bool,required=False)
parser.add_argument('-p','--power',help='Power to raise matching matrix to.',default=1,type=int,required=False)
args = parser.parse_args()

r = args.round
freq = args.frequencyBased
power = args.power
powDict = {'power':power}



# Directories where data and models exist
dataDir = '/mnt/parcellator/parcellation/parcellearning/Data/'

# Directory and file extensions of matching matrices
matchDir = '{}MatchingLibraries/Test/MatchingMatrices/'.format(dataDir)
if freq:
    print 'Using frequency-based matching matrix.\n'
    matchExt = 'MatchingMatrix.0.05.Frequencies.mat'
else:
    print 'Using original matching matrix.\n'
    matchExt = 'MatchingMatrix.0.05.mat'

# Directory and file extension of midline vertices
midsDir = '{}Midlines/'.format(dataDir)
midsExt = 'Midline_Indices.mat'

# Directory and file extension of training objects
testDir = '{}TrainingObjects/FreeSurfer/'.format(dataDir)
testExt = 'TrainingObject.aparc.a2009s.h5'

# Directorys for models and test subject lists
modelDir = '{}Models/'.format(dataDir)
testListDir = '{}TrainTestLists/'.format(dataDir)

# Output directory
outputDir = '{}Predictions/'.format(dataDir)



#### MAPS
methods = ['RandomForest']
exts = ['.p']

methodExtens = ['AtlasSize.1.NumAtlases.Max.Depth.5.NumEst.50']
    
# Maping model type to file extension
classExtsFunc = dict(zip(methods,methodExtens))
# Mapping model type to file name type
classTypeFunc = dict(zip(methods,exts))

# Map file extension to loading functions
loadExt = ['.p','.h5']
loadFuncs = [pickleLoad,load_model]
loadDict = dict(zip(loadExt,loadFuncs))

# Map full hemisphere to abbreviationc
hemispheres = ['Left','Right']
hAbb = ['L','R']
hemiFunc = dict(zip(hemispheres,hAbb))

# Mapping training data type to features included in model
data = ['RestingState','ProbTrackX2','Full']

# List of rounds to process
N = [r]

# Iterate over test sets
for itr in N:
    
    print 'Iteration: {}'.format(itr)
    
    outDirIter = '{}Model_{}/'.format(outputDir,itr)
    
    # Load test subject file, get list of subjects
    testSubjectFile = '{}TestingSubjects.{}.txt'.format(testListDir,itr)
    with open(testSubjectFile,'r') as inFile:
        subjects = inFile.readlines()
    subjects = [x.strip() for x in subjects]

    # Iterate over hemispheres
    for hemi in hemispheres:
        
        print 'Hemisphere: {}'.format(hemi)
        hExt = hemiFunc[hemi]
        
        inMyl = '{}MyelinDensity/285345.{}.MyelinMap.32k_fs_LR.func.gii'.format(dataDir,hExt)
        myl = nb.load(inMyl)
        
        # Iterate over model types (GMM,RandomForest,NetworkModel)
        for classifier in methods:
            
            print 'Classifier: {}'.format(classifier)
            
            # Get classifier file name extension (w.o. data)
            classExt = classExtsFunc[classifier]
            fExt = classTypeFunc[classifier]
            
            for d in data:
                
                print 'Data: {}'.format(d)

                modelBase = '{}.{}.{}.{}.Iteration_{}{}'.format(classifier,
                                  hExt,classExt,d,itr,fExt)
                modelFull = '{}{}'.format(modelDir,modelBase)
                
                if freq:
                    outputExt = '{}.{}.{}.Frequency.Iteration_{}.func.gii'.format(classifier,
                                 hExt,d,itr)
                else:
                    outputExt = '{}.{}.{}.Iteration_{}.func.gii'.format(classifier,
                                 hExt,d,itr)  
                
                G = glob.glob('{}*{}'.format(outDirIter,outputExt))
                
                # check to make sure the given model has not been completed yet
                if len(G) < len(subjects) and os.path.isfile(modelFull):
                
                    currentModel = loadDict[fExt](modelFull)

                    for test_subj in subjects:
                        
                        print 'Subject: {}'.format(test_subj)
                        
                        testOutput = '{}{}.{}'.format(outDirIter,test_subj,outputExt)
                        
                        # Check to make sure current subject hasn't been run yet
                        # If it has, skip
                        if not os.path.isfile(testOutput):
                            
                            testObject = '{}{}.{}.{}'.format(testDir,test_subj,hExt,testExt)                            
                            testMids = '{}{}.{}.{}'.format(midsDir,test_subj,hExt,midsExt)                            
                            testMatch = '{}{}.{}.{}'.format(matchDir,test_subj,hExt,matchExt)
    
                            

                            if fExt == '.p':
                                if classifier == 'RandomForest':
                                    mids = ld.loadMat(testMids)-1
                                    
                                    P = parallelPredictRF(currentModel,testObject,
                                                          testMatch,predictCase = 0)
                                    
                                    P[mids] = 0
                                    myl.darrays[0].data = np.array(P).astype(np.float32)
                                    nb.save(myl,testOutput)
                        else:
                            print '{} already generated.'.format(testOutput)
                else:
                    print '{} for {} already processed.'.format(len(G),outputExt)
                            
        
