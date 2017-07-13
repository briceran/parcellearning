#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:34:27 2017

@author: kristianeschenburg
"""


import argparse

import sys
sys.path.append('..')

import parcellearning.loaded as ld
import parcellearning.classifier_utilities as cu
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import utils
# Import Batch Normalization
from keras.layers.normalization import BatchNormalization

def loadData(subjectList,dataDir,features):
    
    """
    Generates the training data for the neural network.  Assumes that the
    response variable is the last column (i.e. "label" should be the last
    item in "features".)
    """
    
    fExt = 'TrainingObjects/FreeSurfer/'
    mExt = 'Midlines/'
    ext = '.L.TrainingObject.aparc.a2009s.h5'
    
    data = []
    
    for s in subjects:
        
        inTrain = dataDir + fExt + s + ext
        mids = dataDir + mExt + s + '_Midline_Indices.mat'
        
        if os.path.isfile(inTrain) and os.path.isfile(mids):
            
            train = ld.loadH5(inTrain,*['full'])
            train = ld.parseH5(train,features)
            train = train[s]
            
            mergedData = cu.mergeFeatures(train,features)
            
            samples = set(np.arange(mergedData.shape[0]))
            mids = set(ld.loadMat(mids))
            
            coords = np.asarray(list(samples.difference(mids)))
            mergedData = mergedData[coords,:]
            print mergedData.shape
            
            data.append(mergedData)
    
    data = np.row_stack(data)
    
    return data

parser = argparse.ArgumentParser(description='Compute random forest predictions.')
parser.add_argument('-l','--levels', help='Number of levels to include in network.',required=True)
parser.add_argument('-n','--nodes',help='Number of nodes to include in each level.',required=True)
parser.add_argument('-dDir','--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('-f','--features',help='Features to include in model.',required=True)
parser.add_argument('-sl','--subjectList',help='List of subjects to include.',required=True)
args = parser.parse_args()

levels = np.int(args.levels)
nodes = np.int(args.nodes)
dataDir = args.dataDirectory
features = list(args.features.split(','))

print 'Levels: {}'.format(levels)
print 'Nodes: {}'.format(nodes)

subjectFile = args.subjectList
with open(subjectFile,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

trainingData = loadData(subjects,dataDir,features)

xTrain = trainingData[:,:-1]
y = trainingData[:,-1]
oneHotY = utils.to_categorical(y, num_classes=len(set(y)))

samps = xTrain.shape[0]
dims = xTrain.shape[1]

print 'Training data has {} samples, and {} features.'.format(samps,dims)
print 'Building a network with {} hidden layers, each with {} nodes.'.format(levels,nodes)

# instantiate model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=dims))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.30))

c = 0
while c < levels:
    
    print 'Adding layer {}'.format(c+1)
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    c+=1

# we can think of this chunk as the output layer
model.add(Dense(len(set(y)), activation='softmax'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

optim = 'rmsprop'

model.compile(loss='categorical_crossentropy',
              optimizer= optim,
              metrics=['accuracy'])

print 'Model built using {} optimization.  Training now.'.format(optim)

model.fit(xTrain, oneHotY, epochs=50, batch_size=64,verbose=2)
