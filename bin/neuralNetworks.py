#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:34:27 2017

@author: kristianeschenburg
"""

"""
Notes on implementing neural network:

Data is shuffled during training in model.fit if "shuffle" argument is set to true.

Validation is not the same as test data.  Validation data is not used for training or
development of the model -- rather is it used to track the progress of the model 
through loss and accuracy (or other metrics) while the model is being learned.

Test data is held out from both the validation and training data, and used to 
evaluate the model once it has been trained.

We can use the "validation_split" or "validation_data" options as parameters
in model.fit.

"""


import argparse

import sys
sys.path.append('..')

import parcellearning.loaded as ld
import parcellearning.matchingLibraries as lb
import parcellearning.classifier_utilities as cu
import parcellearning.regionalizationMethods as regm

import h5py
import numpy as np
import sklearn
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import utils,optimizers
from keras.optimizers import SGD

# Import Batch Normalization
from keras.layers.normalization import BatchNormalization

###
# Hard coded factors:
    
EVAL_FACTOR = 0.2
DBSCAN_PERC = 0.7

###


## Method to load the training data and aggregate into a single array
def loadData(subjectList,dataDir,features,hemi):
    
    """
    Generates the training data for the neural network.
    
    Parameters:
    - - - - -
        subjectList : list of subjects to include in training set
        dataDir : main directory where data exists -- individual features
                    will exist in sub-directories here
        features : list of features to include
        hemi : hemisphere to process
    """
    
    hemisphere = {}.fromkeys('Left','Right')
    hemisphere['Left'] = 'L'
    hemisphere['Right'] = 'R'
    
    # For now, we hardcode where the data is
    fDir = 'TrainingObjects/FreeSurfer/'
    vDir = 'MatchingLibraries/Test/'
    mDir = 'Midlines/'
    
    trainObjectExt = '.{}.TrainingObject.aparc.a2009s.h5'.format(hemisphere[hemi])
    vLibExt = '.{}.VertexLibrary.Test.p'.format(hemisphere[hemi])
    midExt = '.{}.Midline_Indices.mat'.format(hemisphere[hemi])

    data = {}
    labs = {}
    vlib = {}
    
    dataFeatures = list(set(features).difference({'label'}))
    
    for s in subjectList:

        # Training data
        inTrain = dataDir + fDir + s + trainObjectExt
        # Vertex matching information
        inVLib = dataDir + vDir + s + vLibExt
        # Midline indices
        inMids = dataDir + mDir + s + midExt
        
        # Check to make sure all 3 files exist
        if os.path.isfile(inTrain) and os.path.isfile(inVLib) and os.path.isfile(inMids):
            
            print s
            
            # Load midline indices
            # Subtract 1 for differece between Matlab and Python indexing
            mids = ld.loadMat(inMids)-1
            mids = set(mids)
            
            # Load Vertex Library object and convert to mapping matrix
            vLib = ld.loadPick(inVLib)
            vLibMatrix = lb.buildMappingMatrix(vLib,180)
            

            # Load training data and training labels
            trainH5 = h5py.File(inTrain,mode='r')
            
            # Get unicode ID of subject name
            uni_subj = unicode(s, "utf-8")

            # Get data corresponding to features of interest
            trainFeatures = ld.parseH5(trainH5,dataFeatures)
            trainFeatures = trainFeatures[uni_subj]
            
            # Load Label data
            labelFeatures = ld.parseH5(trainH5,['label'])
            labelFeatures = labelFeatures[uni_subj]
            
            # Merge training features into single array
            mergedDataFeatures = cu.mergeFeatures(trainFeatures,dataFeatures)
            mergedLablFeatures = cu.mergeFeatures(labelFeatures,['label'])
            
            # Get the true data coordiantes (exclude midline coordinates)
            nSamples = set(np.arange(mergedDataFeatures.shape[0]))
            coords = np.asarray(list(nSamples.difference(mids)))

            # Filter the data to include only good data coordiantes
            mergedDataFeatures = mergedDataFeatures[coords,:]
            mergedLablFeatures = mergedLablFeatures[coords,:]
            vLibMatrix = vLibMatrix[coords,1:]
            
            data[s] = mergedDataFeatures
            labs[s] = mergedLablFeatures
            vlib[s] = vLibMatrix
            
            trainH5.close()
    
    # Merged data across all subjects in training sets
    data = aggregateDictValues(data)
    labs = aggregateDictValues(labs)
    vlib = aggregateDictValues(vlib)
    
    return (data,labs,vlib)

def aggregateDictValues(inDict):
    
    """
    Method to aggregate the values of a dictionary, where the values are assumed
    to be numpy arrays.
    """
    
    data = [inDict[k] for k in inDict.keys()]
    data = np.row_stack(data)
    
    return data

def splitDataByLabel(fullDataArray,labelVector):
    
    """
    Get sample data for each label and compile into dictionary, hashed by
    label value.
    
    Parameters:
    - - - - -
        fullDataArray : data array
        labelVector : array of labels
    """
    
    labels = set(np.squeeze(labelVector)).difference({0})
    
    coreData = {}.fromkeys(labels)
    
    for l in labels:
        inds = np.where(labelVector == l)[0]
        coreData[l] = fullDataArray[inds,:]
    
    return coreData

## Functions for Down-sampling data

def noDS(trainingData,labelVector,mm):
    
    """
    Don't apply downsampling
    """
    return (trainingData,labelVector,mm)

def equalDS(trainingData,labelVector,mm):
    
    """
    Apply equal downsampling, where sample number is = min(label samples)
    """
    
    # Get all unique non-zero labels
    labels = set(np.squeeze(labelVector)).difference({0})
    
    # Get samples for each label
    coreTrainData = splitDataByLabel(trainingData,labelVector)
    coreMMData = splitDataByLabel(mm,labelVector)
    # downsample the data
    
    [downTrainData,downMMData] = downsampleRandomly(coreTrainData,coreMMData,labels)
    # build response response vectors for each label
    downResp = cu.buildResponseVector(labels,downTrainData)

    # aggregate samples and response vectors
    equalTrainData = aggregateDictValues(downTrainData)
    equalMMData = aggregateDictValues(downMMData)
    equalResp = aggregateDictValues(downResp)
    
    # return sample and response arrays
    return (equalTrainData,equalMMData,equalResp)

def downsampleRandomly(coreTData,coreMData,labels):
    
    """
    Apply equal downsampling scheme.
    """
    
    # Find sample set with minimum number of points
    m = 10000000
    for k in labels:
        m = np.min([m,coreTData[k].shape[0]])
    
    downTData = {}.fromkeys(labels)
    downMData = {}.fromkeys(labels)
    
    for k in labels:
        dims = coreTData[k].shape[0]
        coords = np.arange(dims)
        tempCoords = np.random.choice(coords,size=m,replace=False)
        
        downTData[k] = coreTData[k][tempCoords,:]
        downMData[k] = coreMData[k][tempCoords,:]
    
    return downTData,downMData

### Need to fix this to incorporate matchingMatrix
def dbsDS(trainingData,labelVector):
    
    """
    Apply DBSCAN to each label sample set, then apply down-sampling.
    """
    
    labels = set(np.squeeze(labelVector)).difference({0})

    # get samples for each label
    coreData = splitDataByLabel(trainingData,labelVector)
    # downsample data for each label using DBSCAN
    dbscanData = regm.trainDBSCAN(coreData,mxp=DBSCAN_PERC)
    
    # apply down-sampling scheme to DBSCAN-ed data
    dbsDataDWS = downsampleRandomly(dbscanData,labels)
    # build response vectors for each label
    dbsRespDWS = cu.buildResponseVector(labels,dbsDataDWS)
    
    # aggregate samples and response vectors
    aggDBS = aggregateDictValues(dbsDataDWS)
    aggResp = aggregateDictValues(dbsRespDWS)
    
    # return samples and response arrays
    return (aggDBS,aggResp)

def shuffleData(training,matching,responses):
    
    [xt,yt] = training.shape
    [xm,ym] = matching.shape
    
    tempData = np.column_stack((training,matching,responses))
    shuffled = sklearn.utils.shuffle(tempData)
    
    trainShuffled = shuffled[:,:yt]
    matchShuffled = shuffled[:,yt:(yt+ym)];
    labelShuffled = shuffled[:,-1]
    
    return (trainShuffled,matchShuffled,labelShuffled)

def computeConstrained(mappingMatrix,x_test,model):
    
    predProb = model.predict_proba(x_test);
    y_test = lb.maximumProbabilityClass(mappingMatrix,predProb)
    
    return y_test

parser = argparse.ArgumentParser(description='Compute random forest predictions.')

parser.add_argument('-dDir','--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('-f','--features',help='Features to include in model.',required=True)
parser.add_argument('-sl','--subjectList',help='List of subjects to include.',required=True)
parser.add_argument('-hm','--hemisphere',help='Hemisphere to proces.',type=str,required=True)

parser.add_argument('-l','--levels', help='Number of levels to include in network.',type=int,default=20)
parser.add_argument('-n','--nodes',help='Number of nodes to include in each level.',type=int,default=100)
parser.add_argument('-e','--epochs',help='Number of epochs.',type=int,default=20)
parser.add_argument('-b','--batchSize',help='Batsh size.',type=int,default=128)

#parser.add_argument('-ns','-numSubj',help='Number of subjects.',type=int,default=30)

parser.add_argument('-opt','--optimizer',help='Optimization scheme.',default='rmsprop',choices=['rmsprop','sgd'])
parser.add_argument('-r','--rate',help='Learning rate.',type=float,default=0.001)

parser.add_argument('-ds','--downSample',help='Type of downsampling to perform.',default='none',
                    choices=['none','equal','dbscan'])

args = parser.parse_args()

ds_funcs = {'none': noDS,
            'equal': equalDS,
            'dbscan': dbsDS}

levels = args.levels
nodes = args.nodes
epochs = args.epochs
batch = args.batchSize
rate = args.rate
hemi = args.hemisphere

optm = args.optimizer
if optm == 'rmsprop':
    opt = optimizers.RMSprop(lr=rate, rho=0.9, epsilon=1e-08, decay=0.0)
else:
    opt = optimizers.SGD(lr=rate, decay=1e-6, momentum=0.9, nesterov=True)


dataDir = args.dataDirectory
features = list(args.features.split(','))

print 'Levels: {}'.format(levels)
print 'Nodes: {}'.format(nodes)
print 'Epochs: {}'.format(epochs)
print 'Batch Size: {}'.format(batch)
print 'Learning rate: {}'.format(rate)

# Load subject data
subjectFile = args.subjectList
with open(subjectFile,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

#ns = np.min([len(subjects),args.ns])
#print 'Number of training subjects: {}'.format(ns)
#subjects = np.random.choice(subjects,size=ns,replace=False)

# Load training data
trainingData,labels,mapMatrix = loadData(subjects,dataDir,features,hemi)
print 'Training Data shape: {}'.format(trainingData.shape)
print 'Label shape: {}'.format(labels.shape)
print 'MapMatrix shape: {}'.format(mapMatrix.shape)

# Down-sample the data using parameters specified by args.downSample
# Currently, only 'equal' works
tempX,tempMM,tempY = ds_funcs[args.downSample](trainingData,labels,mapMatrix)
print 'Training Data shape: {}'.format(tempX.shape)
print 'Label shape: {}'.format(tempY.shape)
print 'MapMatrix shape: {}'.format(tempMM.shape)

# Standardize subject features
S = sklearn.preprocessing.StandardScaler()
training = S.fit_transform(tempX)

# Shuffle features and responses
xTrain,mTrain,yTrain = shuffleData(training,tempMM,tempY)
print 'Training Data shape: {}'.format(xTrain.shape)
print 'Label shape: {}'.format(yTrain.shape)
print 'MapMatrix shape: {}'.format(mTrain.shape)

yTrain = yTrain.astype(np.int32)
yTrain.shape+=(1,)

O = sklearn.preprocessing.OneHotEncoder(sparse=False)
O.fit(yTrain)
OneHotLabels = O.transform(yTrain)

# Dimensions of training data
nSamples = xTrain.shape[0]
input_dim = xTrain.shape[1]
output_dim = OneHotLabels.shape[1]

# Generate one-hot encoded categorical array of response values

print(xTrain.shape)
print(OneHotLabels.shape)
# print(xEval.shape)
# print(yEval.shape)

print 'Training data has {} samples, and {} features.'.format(nSamples,input_dim)
print 'Building a network with {} hidden layers, each with {} nodes.'.format(levels,nodes)

# instantiate model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_dim))
model.add(BatchNormalization())

c = 0
while c < levels:
    
    if c % 2 == 0:

        model.add(Dense(nodes, activation='relu',init='uniform'))
        model.add(BatchNormalization())
    else:
        model.add(Dense(nodes, activation='relu',init='uniform'))
        model.add(BatchNormalization())

    c+=1

# we can think of this chunk as the output layer
model.add(Dense(output_dim, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer= opt,metrics=['accuracy'])

print 'Model built using {} optimization.  Training now.'.format(args.optimizer)

model.fit(xTrain, OneHotLabels, epochs=epochs,
          batch_size=batch,verbose=1,shuffle=True,validation_split=0.2)
