#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:22:13 2017

@author: kristianeschenburg
"""

import copy
import h5py
import numpy as np
import os

import dataUtilities as du
import downsampling as ds
import loaded as ld


"""
##########

Below, we have methods to prepare data for classifiers.

Most of the time, this data will include the feature data, the "true" 
cortical maps, and surface registration information.

We prune the training features to exclude vertices at the midline -- these
vertices do not have any resting-state data association with them.

##########
"""

def buildDataMap(dataDir):
    
    """
    Wrapper to construct data map.
    """
    
    dataMap = {}
    dataMap['object'] = {'{}TrainingObjects/'.format(dataDir) : 
        'TrainingObject.aparc.a2009s.h5'}
    dataMap['midline'] = {'{}Midlines/'.format(dataDir) : 
        'Midline_Indices.mat'}
    dataMap['matching'] = {'{}MatchingLibraries/Test/MatchingMatrices/'.format(dataDir) : 
        'MatchingMatrix.0.05.Frequencies.mat'}
        
    return dataMap


def downsample(inputData,method,L = None):
    
    """
    Wrapper to downsample training data.
    """
    
    methodFuncs = {'equal': ds.byMinimum,
                   'core': ds.byCore}
    
    if not L:
        L = np.arange(1,181)
    else:
        L = np.arange(1,L+1)

    x = inputData[0]
    y = inputData[1]
    m = inputData[2]
    
    [x,y,m] = methodFuncs[method](x,y,m,L)
    
    x = du.mergeValueArrays(x)
    y = du.mergeValueLists(y)
    m = du.mergeValueArrays(m)
    
    return [x,y,m]


def loadData(subjectList,dataMap,features,hemi):
    
    """
    Generates the training data from a list of subjects.
    
    Parameters:
    - - - - -
        subjectList : list of subjects to include in training set
        dataDir : main directory where data exists -- individual features
                    will exist in sub-directories here
        features : list of features to include
        hemi : hemisphere to process
    """

    objDict = dataMap['object'].items()
    objDir = objDict[0][0]
    objExt = objDict[0][1]

    midDict = dataMap['midline'].items()
    midDir = midDict[0][0]
    midExt = midDict[0][1]

    matDict = dataMap['matching'].items()
    matDir = matDict[0][0]
    matExt = matDict[0][1]

    data = {}
    matches = {}

    for s in subjectList:

        # Training data
        trainObject = '{}{}.{}.{}'.format(objDir,s,hemi,objExt)
        midObject = '{}{}.{}.{}'.format(midDir,s,hemi,midExt)
        matObject = '{}{}.{}.{}'.format(matDir,s,hemi,matExt)

        # Check to make sure all 3 files exist
        if os.path.isfile(trainObject) and os.path.isfile(midObject) and os.path.isfile(matObject):

            # Load midline indices
            # Subtract 1 for differece between Matlab and Python indexing
            mids = ld.loadMat(midObject)-1
            mids = set(mids)
            
            match = ld.loadMat(matObject)

            # Load training data and training labels
            trainH5 = h5py.File(trainObject,mode='r')

            # Get data corresponding to features of interest
            subjData = ld.parseH5(trainH5,features)
            trainH5.close()

            nSamples = set(np.arange(subjData[s][features[0]].shape[0]))
            coords = np.asarray(list(nSamples.difference(mids)))
            
            for f in subjData[s].keys():
                tempData = subjData[s][f]
                if tempData.ndim == 1:
                    tempData.shape+=(1,)

                subjData[s][f] = np.squeeze(tempData[coords,:])
                
            match = match[coords,:]
            
            data[s] = subjData[s]
            matches[s] = match

    return [data,matches]


def loadList(subjectFile):
    
    with open(subjectFile,'r') as inFile:
        subjects = inFile.readlines()
    subjects = [x.strip() for x in subjects]
    
    return subjects


def loadUnitaryFeatures(trainingObject):
    
    """
    Load single subject training object.
    
    Parameters:
    - - - - -
        trainingObject : single subject training object
    """

    ID = trainingObject.attrs['ID']

    training = {}
    training[ID] = trainingObject.data
    
    return training



def mapLabelsToData(dataDict,labelDict,labelSet):
    
    """
    Partitions the training data for all subjects by label.
    
    Parameters:
    - - - - -
        dataDict : dictionary mapping subject names to data arrays
        labelDict : dictionary mapping subject names to cortical maps
        labelSet : set of unique labels across all training subjects
    Returns:
    - - - -
        pData : partitioned data
        pLabels : partitioned labels
    """
    
    assert dataDict.keys() == labelDict.keys()
    
    mData = du.mergeValueArrays(dataDict)
    mLabels = du.mergeValueLists(labelDict)

    partData = du.splitArrayByResponse(mData,mLabels,labelSet)

    return partData


def parseKwargs(acceptable,kwargs):
    
    """
    Method to check **kwargs input to methods to receive it.  If some of
    the values are set to None, will remove the key that corresponds to it.
    
    Parameters:
    - - - - -
        classInstance : instance of class whose arguments might needed to be 
                        adapted
        kwargs : possible arguments supplied to function
        
    Returns:
    - - - -
        kwargs : supplied key-value arguments that are included in the 
                    classInstance arguments
    """
    
    output = {}

    if kwargs:
        for key in kwargs.keys():
            
            if key in acceptable:
                output[key] = kwargs[key]

    return output


def vertexMemberships(matchingMatrix,R=180):
        
    """
    Method to partition vertices based on which labels each
    vertex maps to in the training brains.
    Parameters:
    - - - - - 
        matchingMatrix : binary matrix with 1 if 0 maps to vertex, 0 otherwise
        R : number of regions
    Returns:
    - - - - 
        labelVerts : dictionary that maps labels to sets of vertices that 
                        map to it
    """

    labels = np.arange(1,R+1)

    inds = matchingMatrix!=0;
    mm = np.zeros((matchingMatrix.shape))
    mm[inds] = 1;

    idMatrix = mm * labels
    
    labelVerts = {}.fromkeys(list(labels))
    
    for L in labels:
        
        tempColumn = idMatrix[:,L-1]
        inds = np.where(tempColumn == L)[0]
        
        labelVerts[L] = inds
        
    return labelVerts


def matchingPower(match,power):
    
    """
    Raise entries of matching matrix to power.
    """
    
    if power == None:
        match = np.power(match,0)
    elif power == 0:
        nz = np.nonzero(match)
        match[nz] = 1
    else:
        match = np.power(match,power)
        
    return match

def shuffle(inputData):
    
    """
    Given a list of data dictionaries, merge and shuffle the arrays.
    """

    x = inputData[0]
    y = inputData[1]
    m = inputData[2]
    
    N = np.arange(0,x.shape[0])
    np.random.shuffle(N)
    
    x = x[N,:]
    y = y[N]
    m = m[N,:]
    
    return [x,y,m]
    

def validation(inputData,eval_factor):
    
    """
    Processing the validation data from the training set.  The validation 
    data is used to monitor the performance of the model, as the model is 
    trained.  It is expected to withold the validation data 
    from the test data.  The validation is used merely to inform 
    parameter selection.
    
    Parameters:
    - - - - -
        training : list of 3 dictionaries (0 = features, 1 = labels, 2 = matches)
        eval_size : fraction of training size to use as validation set
    """
    
    data = inputData[0]
    labels = inputData[1]
    matches = inputData[2]

    subjects = data.keys()
    
    # By default, will select at least 1 validation subject from list
    full = len(subjects)
    val = max(1,int(np.floor(eval_factor*full)))
    
    print 'Total training subjects: {}'.format(full)
    
    # subject lists for training and validation sets
    train = list(np.random.choice(subjects,size=(full-val),replace=False))
    valid = list(set(subjects).difference(set(train)))
    
    inter = set(train).intersection(set(valid))
    print '{} training, {} validation.'.format(len(train),len(valid))
    print '{} overlap between groups.'.format(len(inter))
    
    training = du.subselectDictionary(train,[data,labels,matches])
    validation = du.subselectDictionary(valid,[data,labels,matches])
    
    validation[0] = du.mergeValueArrays(validation[0])
    validation[1] = du.mergeValueLists(validation[1])
    validation[2] = du.mergeValueArrays(validation[2])

    return [training,validation]


"""
##########

CLASSIFIER PREDICTIONS

##########
"""

# =============================================================================
# def saveClassifier(classifier,output):
#     
#     """
#     Method to save a classifier object.
#     
#     Parameters:
#     - - - - -
#         classifier : for now, just a GMM object of Mahalanobis object.
#     """
#     if classifier._fitted:
#         try:
#             with open(output,"wb") as outFile:
#                 pickle.dump(classifier,outFile,-1);
#         except:
#             pass
#     else:
#         print('Classifier has not been trained.  Not saving.')
# =============================================================================



# =============================================================================
# def standardize(grouped, features):
#     """
#     Method to demean the data from a GroupFeatures object.  This object is
#     just a dictionary of dictionaries -- each main key is a subject ID, with
#     sub-keys correpsonding to features i.e. resting-state, cortical metrics.
# 
#     Standardization is performed upon run-time -- we might want to save the
#     mean and variance of each feature, and will return these, along with a
#     demeaned and unit-varianced GroupFeatures object.
# 
#     Parameters:
#     - - - - -
#         grouped : pickle file -- output of GroupFeatures.save(
#                     {'train': 'outputFile.p'})
#     """
# 
#     if isinstance(grouped, str):
#         trainData = ld.loadH5(grouped, *['full'])        
#     elif isinstance(grouped,h5py._hl.files.File):
#         trainData = grouped        
#     elif isinstance(grouped,dict):
#         trainData = grouped        
#     else:
#         raise ValueError('Training data cannot be empty.')
#         
#     subjects = trainData.keys()
# 
#     mappings = {}.fromkeys(subjects)
#     scalers = {}.fromkeys(features)
# 
#     scale = preprocessing.StandardScaler(with_mean=True,with_std=True)
# 
#     for f in features:
# 
#         c = 0
#         tempData = []
#         scalers[f] = copy.deepcopy(scale)
# 
#         for s in subjects:
#             mappings[s] = {}
# 
#             subjData = trainData[s][f]
#             [x, y] = subjData.shape
#             mappings[s]['b'] = c
#             mappings[s]['e'] = c + x
# 
#             tempData.append(subjData)
#             c += x
# 
#         tempData = np.row_stack(tempData)
#         tempData = scalers[f].fit_transform(tempData)
# 
#         for s in subjects:
#             coords = mappings[s]
#             coordData = tempData[coords['b']:coords['e'],:]
#             trainData[s][f] = coordData
# 
#     return (trainData, scalers)
# =============================================================================



def updatePredictions(storage,members,predictions):
    
    """
    Updates dictionary with predicted labels.  Rather than keeping track of 
    prediction probabilities, keeps track of counts with which vertex is 
    classified as beloning to specific label.
    
    Parameters:
    - - - - -
        storage : array of size N test vertices by K labels to predict
        members : vertices mapping to label
        predictions : predicted labels of members, for 'label' core model
    """
    
    stor = copy.copy(storage)
    stor[members,predictions] += 1

    return stor



# =============================================================================
# 
# def updateScores(storage,label,members,scores):
#         
#         """
#         Updates the dictionary with scores for all vertices that
#         map to label.
#         
#         Parameters:
#         - - - - - 
#             storage : dictionary in which scores are stored
#             label : label of interest
#             members : vertices mapping to label
#             scores : computed scores of members to feature data
#                         of label
#         """
# 
#         for vert,scr in zip(members,scores):
#             # if vertex has not yet been mapped
#             # initialize new key-value pair
#             if not storage[vert]:
#                 storage[vert] = {label: scr}
#             # otherwise update key
#             else:
#                 storage[vert].update({label: scr})
#                 
#         return storage
# 
# =============================================================================
