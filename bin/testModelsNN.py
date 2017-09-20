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
                    
            features : feature to included in the test data numpy array
                        these must be the same as the training data
        """

        nf = []
        for f in features:
            if f != 'label':
                nf.append(f)
        
        print 'Train features: {}'.format(features)
        print 'Test features: {}'.format(nf)
        
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
    
    
def predict(model,mtd,ltvm,mm,**kwargs):
    
    """
    Method to predict neural network model cortical map.
    
    Parameters:
    - - - - -
        model : trained neural network
        
        mtd : member test data in numpy array format
        
        ltvm : label to vertex mappings
        
        mm : matching matrix (binary or frequency)
        
        kwargs : 
    
    """
    
    if kwargs:
        if 'power' in kwargs.keys():
            p = kwargs['power']
        else:
            p = 1
    else:
        p = 1
        
    mm = np.power(mm,p)
    predProbs = model.predict(mtd)
    threshProbs = mm*predProbs[:,1:]
    predicted = np.argmax(threshProbs,axis=1)+1
    
    return predicted

parser = argparse.ArgumentParser(description='Build training objects.')
parser.add_argument('--freq',help='Whether to use frequency-based neighborhood constraint.',
                    default=False,type=bool,required=False)
parser.add_argument('--power',help='Power to raise matching matrix to.',default=1,type=float,required=False)
parser.add_argument('--layers',help='Number of layers in model.',required=True,type=int)
parser.add_argument('--nodes',help='Number of nodes per layer.',required=True,type=int)
parser.add_argument('--testFile',help='List of testing subjects.',required=True,type=str)
parser.add_argument('--modelExtension',help='Extension of model file.',required=True,type=str)
args = parser.parse_args()

freq = args.freq
layers = args.layers
nodes = args.nodes
power = args.power
powDict = {'power':power}

testFile = args.testFile
extension = args.modelExtension

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
outputDir = '{}Predictions/TestReTest/'.format(dataDir)


#### Model parameters as string
modelParams = 'Layers.{}.Nodes.{}.Sampling.equal.Epochs.40.Batch.256.Rate.0.001'.format(layers,nodes)

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

with open(testFile,'r') as inFile:
    testSubjects = inFile.readlines()
testSubjects = [x.strip() for x in testSubjects]


# Iterate over hemispheres
for hemi in hemispheres:

    print 'Hemisphere: {}'.format(hemi)
    hExt = hemiFunc[hemi]

    inMyl = '{}MyelinDensity/285345.{}.MyelinMap.32k_fs_LR.func.gii'.format(dataDir,hExt)

    myl = nb.load(inMyl)

    for d in data:
        
        print 'Data: {}'.format(d)
        
        data_features = dataFeatureFunc[d]
        data_features = list(data_features.split(','))

        modelBase = 'NeuralNetwork.{}.{}.{}.{}.h5'.format(hExt,modelParams,d,extension)
        modelFull = '{}{}'.format(modelDir,modelBase)

        print 'Model: {}'.format(modelFull)
        
        if os.path.isfile(modelFull):
        
            model = load_model(modelFull)
            
            if freq:
                outputExt = 'NeuralNetwork.{}.{}.Frequency.Power.{}.{}.func.gii'.format(hExt,modelParams,freq,power,d,extension)
            else:
                outputExt = 'NeuralNetwork.{}.{}.Binary.{}.{}.func.gii'.format(hExt,modelParams,d,extension)

            G = glob.glob('{}*{}'.format(outputDir,outputExt))

            if len(G) < len(testSubjects):
            
                for test_subj in testSubjects:
                    
                    print 'Subject: {}'.format(test_subj)
                    
                    outputGii = '{}{}.{}'.format(outDirIter,test_subj,outputExt)
                    
                    if not os.path.isfile(outputGii):
                    
                        testObject = '{}{}.{}.{}'.format(testDir,test_subj,hExt,testExt)
                        testMids = '{}{}.{}.{}'.format(midsDir,test_subj,hExt,midsExt)
                        testMatch = '{}{}.{}.{}'.format(matchDir,test_subj,hExt,matchExt)
                        
                        mids = ld.loadMat(testMids)-1

                        [mm,mtd,ltvm] = loadTest(testObject,testMatch,data_features)
                        predicted = predict(model,mtd,ltvm,mm,**powDict)
                        predicted[mids] = 0
                        myl.darrays[0].data = np.array(predicted).astype(np.float32)
                        nb.save(myl,outputGii)
         
                    else:
                        print '{} already generated.'.format(outputGii)
            else:
                print '{} for {} already processed.'.format(len(G),outputExt)

