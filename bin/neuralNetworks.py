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

from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
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
            vLibMatrix = vLibMatrix[coords,:]
            
            data[s] = mergedDataFeatures
            labs[s] = mergedLablFeatures
            vlib[s] = vLibMatrix
            
            trainH5.close()
    
    # Merged data across all subjects in training sets
    data = aggregateDictValues(data)
    labs = aggregateDictValues(labs)
    vlib = aggregateDictValues(vlib)
    
    print 'MatchingMatrix shape: {}'.format(vlib.shape)
    
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

def equalDS(trainingData,mm,labelVector):
    
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
    
    print 'training shape, pre shuffle: {}'.format(training.shape)
    print 'mm shape, pre shuffle: {}'.format(matching.shape)
    print 'label shape, pre shuffle: {}'.format(responses.shape)
    
    [xt,yt] = training.shape

    N = np.arange(xt);
    N = sklearn.utils.shuffle(N);
    
    trainShuffled = training[N,:]
    matchShuffled = matching[N,:]
    labelShuffled = responses[N]
    
    print 'training shape, post shuffle: {}'.format(trainShuffled.shape)
    print 'mm shape, post shuffle: {}'.format(matchShuffled.shape)
    print 'label shape, post shuffle: {}'.format(labelShuffled.shape)

    return (trainShuffled,matchShuffled,labelShuffled)


class TestCallback(callbacks.Callback):
    
    """
    Callback to test neighborhood constrained accuracy computations.
    """
    def __init__(self, mappingMatrix, test_x,test_y):

        self.mm = mappingMatrix
        self.x_test = test_x
        self.y_test = test_y

    def on_epoch_end(self, epoch, logs={}):
        
        x = self.x_test
        y = self.y_test
        mm = self.mm
        
        print '\n'

        predProb = self.model.predict_proba(x)

        threshed = mm*predProb;
        y_pred = np.argmax(threshed,axis=1)
        
        print 'Minimum true class: {}'.format(np.min(y))
        print 'Minimum pred class: {}'.format(np.min(y_pred))
        
        print 'y_true: {}'.format(y)
        print 'y_pred: {}'.format(y_pred)

        loss,_ = self.model.evaluate(x, y, verbose=0)
        acc = np.mean(y == y_pred)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


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


# Down-sample the data using parameters specified by args.downSample
# Currently, only 'equal' works
tempX,tempM,tempY = ds_funcs[args.downSample](trainingData,mapMatrix,labels)


# Standardize subject features
S = sklearn.preprocessing.StandardScaler()
trainTransformed = S.fit_transform(tempX)

# Shuffle features and responses
xTrain,mTrain,yTrain = shuffleData(trainTransformed,tempM,tempY)


yTrain = yTrain.astype(np.int32)
print yTrain.shape

O = sklearn.preprocessing.OneHotEncoder(sparse=False)
O.fit(yTrain)
OneHotLabels = O.transform(yTrain)

# Dimensions of training data
nSamples = xTrain.shape[0]
input_dim = xTrain.shape[1]
output_dim = OneHotLabels.shape[1]


# Generate validation data set
N = np.arange(nSamples);
dSamples = int(np.floor(nSamples*EVAL_FACTOR))
evals_coords = np.random.choice(N, size=(dSamples,), replace=False)
train_coords = np.asarray(list(set(N).difference(set(evals_coords))))

eval_x = xTrain[evals_coords,:]
eval_y = OneHotLabels[evals_coords,:]
flat_y = np.argmax(eval_y,axis=1)

eval_m = mTrain[evals_coords,:]

train_x = xTrain[train_coords,:]
train_y = OneHotLabels[train_coords,:]
train_m = mTrain[train_coords,:]

# final dimensionf of data
nSamples = train_x.shape[0]

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

model.fit(train_x, train_y, epochs=epochs,
          batch_size=batch,verbose=1,shuffle=True,
          callbacks=[TestCallback(eval_m,eval_x,flat_y)])
