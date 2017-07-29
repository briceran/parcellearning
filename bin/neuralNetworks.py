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
import time

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
import pickle

from keras import callbacks, optimizers
from keras.models import Sequential
from keras.layers import Dense,normalization


###
# Hard coded factors:
    
# default validation set size (fraction of training set)
EVAL_FACTOR = 0.2

# Default DBSCAN fraction of kept points
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
    
    H = hemisphere[hemi]
    
    # For now, we hardcode where the data is
    fDir = 'TrainingObjects/FreeSurfer/'
    vDir = 'MatchingLibraries/Test/MatchingMatrices/'
    mDir = 'Midlines/'
    
    trainObjectExt = '.{}.TrainingObject.aparc.a2009s.h5'.format(H)
    vLibExt = '.{}.MatchingMatrix.0.05.mat'.format(H)
    midExt = '.{}.Midline_Indices.mat'.format(H)

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

            # Load midline indices
            # Subtract 1 for differece between Matlab and Python indexing
            mids = ld.loadMat(inMids)-1
            mids = set(mids)
            
            # Load MatchingMatrix object
            vLibMatrix = ld.loadMat(inVLib)            

            # Load training data and training labels
            trainH5 = h5py.File(inTrain,mode='r')
            
            # Get unicode ID of subject name
            uni_subj = unicode(s, "utf-8")

            # Get data corresponding to features of interest
            trainFeatures = ld.parseH5(trainH5,dataFeatures)
            try:
                trainFeatures = trainFeatures[uni_subj]
            except:
                trainFeatures = trainFeatures[s]
            
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
def dbsDS(trainingData,mm,labelVector):
    
    """
    Apply DBSCAN to each label sample set, then apply down-sampling.
    """
    
    labels = set(np.squeeze(labelVector)).difference({0})

    # get samples for each label
    coreData = splitDataByLabel(trainingData,labelVector)
    coreMM = splitDataByLabel(mm,labelVector)
    
    # Compute filtered coordinates using DBSCAN
    clusterCoordinates = regm.trainDBSCAN(coreData,mxp=DBSCAN_PERC)
    
    # Extract the samples that passed using DBSCAN filtering
    dbscanData = condenseDBSCAN(coreData,clusterCoordinates)
    dbscanMM = condenseDBSCAN(coreMM,clusterCoordinates)

    # Apply random down-sampling scheme to DBSCAN-ed data and matchingMatrix
    # so that all labels match the smallest label
    [dataDS,mmDS] = downsampleRandomly(dbscanData,dbscanMM,labels)

    # Build response vectors for each label
    dbsRespDS = cu.buildResponseVector(labels,dataDS)
    
    # Aggregate samples and response vectors
    aggDataDS= aggregateDictValues(dataDS)
    aggMMDS = aggregateDictValues(mmDS)
    aggRespDS = aggregateDictValues(dbsRespDS)
    
    # return samples and response arrays
    return (aggDataDS,aggMMDS,aggRespDS)

def condenseDBSCAN(inSamples,coords):
    
    """
    Given the coordinates of dbscan-accepted samples, reduce the sample
    dimensionality.
    """
    
    outSamples = {}.fromkeys(coords.keys())
    
    for k in outSamples.keys():
        outSamples[k] = inSamples[k][coords[k],:]
        
    return outSamples

def shuffleData(training,matching,responses):

    [xt,yt] = training.shape

    N = np.arange(xt);
    N = sklearn.utils.shuffle(N);
    
    trainShuffled = training[N,:]
    matchShuffled = matching[N,:]
    labelShuffled = responses[N]

    return (trainShuffled,matchShuffled,labelShuffled)


class ConstrainedCallback(callbacks.Callback):
    
    """
    Callback to test neighborhood-constrained accuracy computations.
    
    Parameters:
    - - - - -
        mappingMatrix : binary matrix of mapping results, where each row is a
                        sample, and each column is a label.  If a sample
                        mapped to a label during surface registration, the
                        index [sample,label] = 1.
                        
        test_x : validation data feature matrix
        
        y_true : validation data response vector
        
        y_oneHot : validation data one-hot matrix
        
        metricKeys : if we wish to save the loss and accuracy using the 
                    contsrained method, provide key names in format 
                    [lossKeyName, accKeyName] to save these values in a 
                    dictionary
                    
        
    The Keras model required one-hot arrays for evaluation of the model --
    it does not accept multi-valued integer vectors.
        
    """
    def __init__(self, mappingMatrix, test_x, y_true, y_oneHot,metricKeys):

        self.mm = mappingMatrix
        self.x_test = test_x
        
        self.y_true = y_true
        self.y_oneHot = y_oneHot
        self.metricKeys = metricKeys
        
        self.metrics = {k: [] for k in metricKeys}

    def on_epoch_end(self, epoch, logs={}):
        
        x = self.x_test
        y_true = self.y_true
        y_oneHot = self.y_oneHot
        mm = self.mm

        # Compute the prediction probability of all samples for all classes.
        # N x K matrix
        predProb = self.model.predict_proba(x)

        # Include only those prediction probabilities for classes that the 
        # samples mapped to during surface registration
        threshed = mm*predProb;
        
        # Find the class with the greatest prediction probability
        y_pred = np.argmax(threshed,axis=1)

        # Evalute the loss and accuracy of the model
        loss,_ = self.model.evaluate(x, y_oneHot, verbose=0)
        acc = np.mean(y_true == y_pred)
        
        self.metrics[self.metricKeys[0]].append(loss)
        self.metrics[self.metricKeys[1]].append(acc)
        
        print('\n{}: {}, {}: {}\n'.format(self.metricKeys[0],loss,self.metricKeys[1],acc))




################################################
################################################
################################################
        
# Begin scipt.


parser = argparse.ArgumentParser(description='Compute random forest predictions.')

# Parameters for input data
parser.add_argument('-dDir','--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('-f','--features',help='Features to include in model.',required=True)
parser.add_argument('-sl','--subjectList',help='List of subjects to include.',required=True)
parser.add_argument('-hm','--hemisphere',help='Hemisphere to proces.',required=True)
parser.add_argument('-o','--output',help='Name of file storing training model',required=True)

parser.add_argument('-ds','--downSample',help='Type of downsampling to perform.',default='none',
                    choices=['none','equal','dbscan'])


# Parameters for network architecture
parser.add_argument('-l','--levels', help='Number of levels to include in network.',
                    type=int,default=20,required=False)
parser.add_argument('-n','--nodes',help='Number of nodes to include in each level.',
                    type=int,default=100,required=False)
parser.add_argument('-e','--epochs',help='Number of epochs.',type=int,
                    default=20,required=False)
parser.add_argument('-b','--batchSize',help='Batch size.',type=int,
                    default=128,required=False)

# Parameters for weight updates
parser.add_argument('-opt','--optimizer',help='Optimization scheme.',type=str,
                    default='rmsprop',choices=['rmsprop','sgd'],required=False)
parser.add_argument('-r','--rate',help='Learning rate.',type=float,
                    default=0.001,required=False)

args = parser.parse_args()

ds_funcs = {'none': noDS,
            'equal': equalDS,
            'dbscan': dbsDS}

# Brain hemisphere
hemi = args.hemisphere

# Number of hidden layers
levels = args.levels
# Number of nodes per hidden layer
nodes = args.nodes
# Number of training epochs
epochs = args.epochs
# Size of each training batch
batch = args.batchSize
# learning rate
rate = args.rate




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
print 'Loading subject data.'
now = time.time()
trainingData,labels,mapMatrix = loadData(subjects,dataDir,features,hemi)
later = time.time()
print 'Loaded in {} seconds.\n'.format(int(later-now))


# Down-sample the data using parameters specified by args.downSample
# Currently, only 'equal' works
print 'Applying {} sample reduction.'.format(args.downSample)
now = time.time()
tempX,tempM,tempY = ds_funcs[args.downSample](trainingData,mapMatrix,labels)
later = time.time()
print 'Reduced in {} seconds.\n'.format(int(later-now))


# Standardize subject features
print 'Standardizing.\n'
S = sklearn.preprocessing.StandardScaler()
trainTransformed = S.fit_transform(tempX)

# Shuffle features and responses
print 'Shuffling.\n'
xTrain,mTrain,yTrain = shuffleData(trainTransformed,tempM,tempY)


yTrain = yTrain.astype(np.int32)
yTrain.shape+=(1,)

O = sklearn.preprocessing.OneHotEncoder(sparse=False)
O.fit(yTrain.reshape(-1,1))
OneHotLabels = O.transform(yTrain.reshape(-1,1))

# Dimensions of training data
nSamples = xTrain.shape[0]
input_dim = xTrain.shape[1]


# Generate validation data set
print 'Generating validation set.\n'
N = np.arange(nSamples);
dSamples = int(np.floor(nSamples*EVAL_FACTOR))
evals_coords = np.random.choice(N, size=(dSamples,), replace=False)
train_coords = np.asarray(list(set(N).difference(set(evals_coords))))

eval_x = xTrain[evals_coords,:]
eval_y = OneHotLabels[evals_coords,:]
flat_eval_y = np.argmax(eval_y,axis=1)
eval_m = mTrain[evals_coords,:]

train_x = xTrain[train_coords,:]
train_y = OneHotLabels[train_coords,:]
flat_train_y = np.argmax(train_y,axis=1)
train_m = mTrain[train_coords,:]

# final dimensionf of data
nSamples = train_x.shape[0]
output_dim = train_y.shape[1]

print 'Training data has {} samples, and {} features.'.format(nSamples,input_dim)
print 'Building a network with {} hidden layers, each with {} nodes.\n'.format(levels,nodes)

# instantiate model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_dim))
model.add(normalization.BatchNormalization())

c = 0
while c < levels:
    
    # in case you want to add alternating activation functions
    # currently, all layers are relu
    
    model.add(Dense(nodes,activation='relu',init='uniform'))
    model.add(normalization.BatchNormalization())

    c+=1

# we can think of this chunk as the output layer
model.add(Dense(output_dim, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer= opt,metrics=['accuracy'])

print 'Model built using {} optimization.  Training now.\n'.format(args.optimizer)


ConstrainedTE = ConstrainedCallback(eval_m,eval_x,flat_eval_y,eval_y,['consTestLoss','consTestAcc'])
ConstrainedTR = ConstrainedCallback(train_m,train_x,flat_train_y,train_y,['consTrainLoss','consTrainAcc'])

history = model.fit(train_x, train_y, epochs=epochs,
          batch_size=batch,verbose=2,shuffle=True,
          callbacks=[ConstrainedTE,ConstrainedTR])

outTrained = args.output
outTrainedFile = ''.join([outTrained,'.h5'])
outHistoryFile = ''.join([outTrained,'_History.p'])

model.save(outTrainedFile)

modelHistory = history.history
fullHistory = dict(modelHistory.items() + ConstrainedTE.metrics.items() + ConstrainedTR.metrics.items())

with open(outHistoryFile,'w') as outFile:
    pickle.dump(fullHistory,outFile,-1)