#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:22:13 2017

@author: kristianeschenburg
"""

import os
import sys

import dataUtilities as du
import loaded as ld
import matchingLibraries as lb

import copy
import h5py
import pickle

import numpy as np
import scipy.io as sio
from sklearn import preprocessing

"""
##########

DOWNSAMPLING

##########
"""


        



def downsampleByMinimum(trainingData,trainingLabels,trainingMatches,labelSet):
    
    """
    Downsamples the training data so that the data for each label has the same
    number of samples.  Finds the label that has the minimum number of samples,
    and random selects the same number of samples from the remaining labels.
    
    Generally, we use this approach for neural network training data, after
    having aggregated all of the training data.  If we were to downsample on
    a subject by subject basis, we would have very little training data per 
    subject and would lose discriminative power.
    
    Parameters:
    - - - - -
        trainingData : dictionary of training data, where keys are subjects
                        and values are the vertex-wise data arrays
        trainingLabels : dictionary of training labels, where keys are subjects
                            and values are vertex-wise label assignments
        trainingMatches : dictionary of matches, where keys are subjects, and
                            values are vertex-to-label frequency arrays
        labelSet : set of unique labels across all training subjects
    Returns:
    - - - -
        data : downsampled data array
        labels : downsampled response vectors
        matches : downsampled matches
        coreMaps : indices of "good" vertices, per subject, per label
    """
    
    minSize = sys.maxint
    
    pData = mapLabelsToData(trainingData,trainingLabels,labelSet)
    pMatches = mapLabelsToData(trainingMatches,trainingLabels,labelSet)
    
    pLabels = du.buildResponseVector(pData)
    
    for L in labelSet:
        
        tempData = pData[L]
        minSize = min(minSize,tempData.shape[0])
    
    for L in labelSet:
        
        tempData = pData[L]
        tempMatches = pMatches[L]
        tempLabels = pData[L]
        
        inds = np.random.choice(np.arange(tempData.shape[0],size=minSize,
                                          replace=False))
        pData[L] = tempData[inds,:]
        pMatches[L] = tempMatches[inds,:]
        pLabels[L] = tempLabels[inds,:]
        
    return [pData,pMatches,pMatches]

"""
##########

Below, we have methods to help build generic atlases and classifiers.

##########
"""


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
    
    supraData = du.mergeValueArrays(dataDict)
    supraLabels = du.mergeValueLists(labelDict)

    partData = du.splitArrayByResponse(supraData,supraLabels,labelSet)
    partResp = du.buildResponseVector(partData)
    
    if not compareTrainingDataSize(partData,partResp):
        raise ValueError('Training data is flawed.')
    
    return partData


def compareTrainingDataSize(partitionedData,partitionedResponse):
    
    """
    Method to ensure that the length of the response vector is the same 
    length as the number of observations in the training feature data.
    
    This must be true in order to actually train the classifiers for each
    label.
    """
    cond = True

    for f,r in zip(set(partitionedData.keys()),set(partitionedResponse.keys())):
        
        sf = partitionedData[f].shape[0]
        sr = partitionedResponse[r].shape[0]
        
        if sf != sr:
            cond = False
    
    return cond


def getLabels(trainData):
        
        """
        Get all unique labels that exist across entire set of training data
        
        Parameters:
        - - - - - 
            trainData : training data object
        Returns:
        - - - - 
            labs : list of all the unique label values in a training object
            
        """
        
        supraLabels = du.mergeValueLists(labelDict)
        
        
        
        labs = set([])
        
        # loop over all training subjects in training data
        # by construction, will have a feature called 'label'
        for subj in trainData:

            labelData = np.squeeze(trainData[subj]['label'])
            labs.update(set(labelData))
            
        return list(labs)

    
def mergeLabelData(labelData,responseData,labels):
    
    """
    Method to merge the training data corresponding to a set of labels.
    
    Parameters:
    - - - - - 
        labelData : dictionary where keys correspond to unique
                    labels and values correspond to aggregate data for that 
                    label across all training subjects      
        responseData : dictionary where keys correspond to unique labels and 
                        values correspond to vectos of length N_i samples
                        (for label i)
        labels : labels to aggregate over
        
    Returns:
    - - - -
        learnData : array correspond to "stacked" labelData arrays, where only
                    data corresponding to l in labels are stacked
        y : array corresponding to "stacked" response vectors, where only 
            vectors corresponding to l in labels are stacked
    """
    
    learnData = []    
    y = []
    
    for l in labels:
        if l in labels:
            learnData.append(labelData[l])
            y.append(responseData[l])
        
    learnData = np.row_stack(learnData)
    y = np.row_stack(y)
    
    return(learnData,y)

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


def prepareUnitaryFeatures(trainingObject):
    
    """
    Parameters:
    - - - - -
        trainingObject : single subject training object
    """

    ID = trainingObject.attrs['ID']

    training = {}
    training[ID] = trainingObject.data
    
    return training

def vertexMemberships(matchingMatrix,R):
        
    """
    Method to partition vertices based on which labels each
    vertex maps to in the training brains.  As more training brains are
    included in the **MatchingFeaturesTest** object, a vertex might map
    to an increasing variety of labels.
    
    We do this so we can apply a classifier to multiple vertices at the same
    time, rather than looping through each vertex, through all classifiers.
    
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

def standardize(grouped, features):
    """
    Method to demean the data from a GroupFeatures object.  This object is
    just a dictionary of dictionaries -- each main key is a subject ID, with
    sub-keys correpsonding to features i.e. resting-state, cortical metrics.

    Standardization is performed upon run-time -- we might want to save the
    mean and variance of each feature, and will return these, along with a
    demeaned and unit-varianced GroupFeatures object.

    Parameters:
    - - - - -
        grouped : pickle file -- output of GroupFeatures.save(
                    {'train': 'outputFile.p'})
    """

    if isinstance(grouped, str):
        trainData = ld.loadH5(grouped, *['full'])        
    elif isinstance(grouped,h5py._hl.files.File):
        trainData = grouped        
    elif isinstance(grouped,dict):
        trainData = grouped        
    else:
        raise ValueError('Training data cannot be empty.')
        
    subjects = trainData.keys()

    mappings = {}.fromkeys(subjects)
    scalers = {}.fromkeys(features)

    scale = preprocessing.StandardScaler(with_mean=True,with_std=True)

    for f in features:

        c = 0
        tempData = []
        scalers[f] = copy.deepcopy(scale)

        for s in subjects:
            mappings[s] = {}

            subjData = trainData[s][f]
            [x, y] = subjData.shape
            mappings[s]['b'] = c
            mappings[s]['e'] = c + x

            tempData.append(subjData)
            c += x

        tempData = np.row_stack(tempData)
        tempData = scalers[f].fit_transform(tempData)

        for s in subjects:
            coords = mappings[s]
            coordData = tempData[coords['b']:coords['e'],:]
            trainData[s][f] = coordData

    return (trainData, scalers)

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

def updateScores(storage,label,members,scores):
        
        """
        Updates the dictionary with scores for all vertices that
        map to label.
        
        Parameters:
        - - - - - 
            storage : dictionary in which scores are stored
            label : label of interest
            members : vertices mapping to label
            scores : computed scores of members to feature data
                        of label
        """

        for vert,scr in zip(members,scores):
            # if vertex has not yet been mapped
            # initialize new key-value pair
            if not storage[vert]:
                storage[vert] = {label: scr}
            # otherwise update key
            else:
                storage[vert].update({label: scr})
                
        return storage
