#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:34:08 2017

@author: kristianeschenburg
"""

import sys
sys.path.append('..')

import parcellearning.classifier_utilities as cu
import parcellearning.loaded as ld
import parcellearning.malp as malp
import parcellearning.MixtureModel as MM

from keras.models import load_model

import pickle
import nibabel as nb
import numpy as np


def pickleLoad(inFile):
    
    with open(inFile,'r') as inputFile:
        data = pickle.load(inputFile)
    
    return data

def loadTest(y,yMatch,features):
        
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
        features = list(features.split(','))
        
        # load test subject data, save as attribtues
        tObject = ld.loadH5(y,*['full'])
        ID = tObject.attrs['ID']
        
        parsedData = ld.parseH5(tObject,features)
        tObject.close()

        data = parsedData[ID]
        mtd = cu.mergeFeatures(data,features)

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
#methods = ['GMM','RandomForest','NetworkModel']
#exts = ['.p','.p','.h5']

methods = ['GMM','NetworkModel']
exts = ['.p','.h5']

#methodExtens = ['Covariance.diag.NumComponents.2',
#              'AtlasSize.1.NumAtlases.Max.Depth.5.NumEst.50',
#              'Layers.3.Nodes.1250.Sampling.equal.Epochs.60.Batch.256.Rate.0.001']

methodExtens = ['Covariance.diag.NumComponents.2',
                'Layers.3.Nodes.1250.Sampling.equal.Epochs.60.Batch.256.Rate.0.001']

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
dataFeatures = ['fs_cort,fs_subcort,sulcal,myelin,curv,label',
                'pt_cort,pt_subcort,sulcal,myelin,curv,label',
                'fs_cort,fs_subcort,pt_cort,pt_subcort,sulcal,myelin,curv,label']

dataFeatureFunc = dict(zip(data,dataFeatures))

# Number of testing sets
N = 10

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
                print 'Model Name: {}'.format(modelFull)
                
                currentModel = loadDict[fExt](modelFull)
                
                outputExt = '{}.{}.{}.Iteration_{}.func.gii'.format(classifier,
                             hExt,d,itr)    
                
                for test_subj in subjects:
                    
                    print 'Subjects: {}'.format(test_subj)
                    
                    testObject = '{}{}.{}.{}'.format(testDir,test_subj,hExt,testExt)
                    print 'Test Object: {}'.format(testObject)
                    
                    testMids = '{}{}.{}.{}'.format(midsDir,test_subj,hExt,midsExt)
                    print 'Test Mids: {}'.format(testMids)
                    
                    testMatch = '{}{}.{}.{}'.format(matchDir,test_subj,hExt,matchExt)
                    print 'Test Match: {}'.format(testMatch)
                    
                    testOutput = '{}{}.{}'.format(outDirIter,test_subj,outputExt)
                    print 'Test Output: {}'.format(testOutput)
                    
                    mids = ld.loadMat(testMids)-1
    
                    if fExt == '.p':
                        # If model was a random forest,current model is a LIST
                        # of models.  We feed this in to malp.parallelPredictiong
                        # along with the test data
                        if classifier == 'RandomForest':
                            modelPreds = []
                            for model in currentModel:
                                model.predict(testObject,testMatch,testMids)
                                modelPreds.append(model.predicted)
                            
                            modelPreds = np.column_stack(modelPreds)
                            predicted = []
                            for i in np.arange(modelPreds.shape[0]):
        
                                L = list(modelPreds[i,:])
                                maxProb = max(set(L),key=L.count)
                                predicted.append(maxProb)
                            
                            predicted = np.asarray(predicted)

                        elif classifier == 'GMM':
                            currentModel.predict(testObject,testMatch,testMids)
                            
                        predicted = currentModel.predicted
                    
                    elif fExt == '.h5':
                        features = dataFeatureFunc[classifier]
                        
                        [threshed,mtd,_] = loadTest(testObject,testMatch,)
                        mtd[mids,:] = 0
                        threshed[mids,:] = 0
                        
                        predProbs = currentModel.predict(mtd)
                        threshProbs = threshed*predProbs
                        
                        predicted = np.argmax(threshProbs,axis=1)+1
                    
                    predicted[mids] = 0
                    
                    print 'Predicted shape: {}'.format(predicted.shape)
                    
                    myl.darrays[0].data = np.array(predicted).astype(np.float32)
                    nb.save(myl,testOutput)
                    
                
                
                
                
        
        