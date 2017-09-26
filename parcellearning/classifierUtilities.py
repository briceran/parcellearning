#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:22:13 2017

@author: kristianeschenburg
"""

import copy
import numpy as np
import dataUtilities as du


"""
##########

Below, we have methods to prepare data for classifiers.

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
