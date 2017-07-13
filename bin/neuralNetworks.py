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
import sklearn
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
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
            
            data.append(mergedData)
    
    data = np.row_stack(data)
    
    return data

parser = argparse.ArgumentParser(description='Compute random forest predictions.')

parser.add_argument('-dDir','--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('-f','--features',help='Features to include in model.',required=True)
parser.add_argument('-sl','--subjectList',help='List of subjects to include.',required=True)

parser.add_argument('-l','--levels', help='Number of levels to include in network.',type=int,default=10)
parser.add_argument('-n','--nodes',help='Number of nodes to include in each level.',type=int,default=5)
parser.add_argument('-e','--epochs',help='Number of epochs.',type=int,default=20)
parser.add_argument('-b','--batchSize',help='Batsh size.',type=int,default=128)

parser.add_argument('-ns','-numSubj',help='Number of subjects.',type=int,default=30)

parser.add_argument('-opt','--optimizer',help='Optimization scheme.',default='rmsprop',choices=['rmsprop','sgd'])
parser.add_argument('-r','--rate',help='Learning rate.',type=float,default=0.001)
args = parser.parse_args()


optm = args.optimizer
if optm == 'rmsprop':
    opt = 'rmsprop'
    rate = args.rate
else:
    opt = SGD(lr=rate, decay=1e-6, momentum=0.9, nesterov=True)
    rate = 0.01

levels = args.levels
nodes = args.nodes
epochs = args.epochs
batch = args.batchSize

dataDir = args.dataDirectory
features = list(args.features.split(','))

print 'Levels: {}'.format(levels)
print 'Nodes: {}'.format(nodes)
print 'Epochs: {}'.format(epochs)
print 'Batch Size: {}'.format(batch)

# Load subject data
subjectFile = args.subjectList
with open(subjectFile,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

ns = np.min([len(subjects),args.ns])
print 'Number of training subjects: {}'.format(ns)
subjects = np.random.choice(subjects,size=ns,replace=False)

trainingData = loadData(subjects,dataDir,features)
trainingData = sklearn.utils.shuffle(trainingData)

# Standardize subject features
S = sklearn.preprocessing.StandardScaler()
training = S.fit_transform(trainingData)

# Get training features and responses
xTrain = training[:,:-1]
y = training[:,-1].astype(np.int32)

oneHotY = utils.to_categorical(y, num_classes=len(set(y))+1)
oneHotY = oneHotY[:,1:]

# Dimensions of training data
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
    
    if c % 2 == 0:

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    else:
        model.add(Dense(128, activation='sigmoid'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    
    c+=1

# we can think of this chunk as the output layer
model.add(Dense(len(set(y)), activation='softmax'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=['accuracy'])

print 'Model built using {} optimization.  Training now.'.format(args.optimizer)

model.fit(xTrain, oneHotY, epochs=epochs, batch_size=batch, verbose=1)
