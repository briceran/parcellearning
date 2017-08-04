#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:34:08 2017

@author: kristianeschenburg
"""

import sys
sys.path.append('..')

import copy

import parcellearning.classifier_utilities as cu
import parcellearning.loaded as ld

from keras.models import load_model

import glob
import os
import pickle
import nibabel as nb
import numpy as np

def pickleLoad(inFile):
    
    with open(inFile,'r') as inputFile:
        data = pickle.load(inputFile)
    
    return data

def loadTest(yObject,yMatch,features):
        
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
        for f in features:
            if f != 'label':
                nf.append(f)
        
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

# Directories where data and models exist
dataDir = '/mnt/parcellator/parcellation/parcellearning/Data/'

# Directory and file extensions of matching matrices
matchDir = '{}MatchingLibraries/Test/MatchingMatrices/'.format(dataDir)
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
# Mapping model type to file name type and file name extension

methods = ['NeuralNetwork']
exts = ['.h5']

methodExtens = ['Layers.3.Nodes.100.Sampling.equal.Epochs.30.Batch.256.Rate.0.001']

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
dataFeatures = ['fs_cort,fs_subcort,sulcal,myelin,curv',
                'pt_cort,pt_subcort,sulcal,myelin,curv',
                'fs_cort,fs_subcort,pt_cort,pt_subcort,sulcal,myelin,curv']

dataFeatureFunc = dict(zip(data,dataFeatures))

# Number of testing sets
N = 1

# Iterate over test sets
for itr in np.arange(N):
    
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
                
                data_features = dataFeatureFunc[d]
                
                modelBase = '{}.{}.{}.{}.Iteration_{}{}'.format(classifier,
                                  hExt,classExt,d,itr,fExt)
                modelFull = '{}{}'.format(modelDir,modelBase)
                
                if os.path.isfile(modelFull):
                
                    currentModel = loadDict[fExt](modelFull)
                    
                    outputExt = '{}.{}.{}.Iteration_{}.func.gii'.format(classifier,
                                 hExt,d,itr)
                    
                    G = glob.glob('{}*{}'.format(outDirIter,outputExt)) 
                    if len(G) < len(subjects):
                    
                        for test_subj in subjects:
                            
                            print 'Subject: {}'.format(test_subj)
                            
                            testOutput = '{}{}.{}'.format(outDirIter,test_subj,outputExt)
                            
                            if not os.path.isfile(testOutput):
                            
                                testObject = '{}{}.{}.{}'.format(testDir,test_subj,hExt,testExt)
                                testMids = '{}{}.{}.{}'.format(midsDir,test_subj,hExt,midsExt)
                                testMatch = '{}{}.{}.{}'.format(matchDir,test_subj,hExt,matchExt)
                                
                                mids = ld.loadMat(testMids)-1
                                #mids = ld.loadMat(testMids)
                
                                if fExt == '.h5':
            
                                    [threshed,mtd,_] = loadTest(testObject,testMatch,data_features)
                                    #mtd[mids,:] = 0
                                    print 'mtd shape: {}'.format(mtd.shape)
                                    #threshed[mids,:] = 0
                                    print 'threshed shape: {}'.format(threshed.shape)
                                    
                                    predProbs = currentModel.predict(mtd)
                                    print 'pp shape: {}'.format(predProbs.shape)
                                    threshProbs = threshed*predProbs[:,1:]
                                    
                                    predicted = np.argmax(threshProbs,axis=1)+1
                                
                                predicted[mids] = 0
            
                                myl.darrays[0].data = np.array(predicted).astype(np.float32)
                                nb.save(myl,testOutput)
                                                
                            else:
                                print '{} already generated.'.format(testOutput)
                    else:
                        print '{} for {} already processed.'.format(len(G),outputExt)
        
