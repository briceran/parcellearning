#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:07:37 2017

@author: kristianeschenburg
"""

import copy
import numpy as np

def downsampleByCore(trainingData,trainingLabels,trainingMatches,labelSet):
    
    """
    Downsample the training data for each subject, for each label, by
    selecting the core vertices for each label.  We can follow this up by
    downsampling by the minimum label size, or we can concatenate the data 
    into a single array and feed this into a neural network.
    
    Likewise, we can partition this data by label, and feed the partitioned
    data into a Random Forest, GMM, or MALP classifier.
    
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
    
    assert trainingData.keys() == trainingLabels.keys()
    
    data = copy.deepcopy(trainingData)
    labels = copy.deepcopy(trainingLabels)
    matches = copy.deepcopy(trainingMatches)
    
    subjects = trainingData.keys()
    coreMaps = {}.fromkeys(subjects)

    for subj in trainingData.keys():
        
        tempData = trainingData[subj]
        tempLabel = trainingLabels[subj]
        tempMatch = trainingMatches[subj]

        print 'Matching shape: {}'.format(tempMatch.shape)
        print 'Training data shape: {}'.format(tempData.shape)

        cores = labelCores(tempMatch,0.7)
        coreMaps[subj] = cores
        
        cores = np.squeeze(np.concatenate(cores.values())).astype(np.int32)
        cores = np.sort(cores)

        data[subj] = tempData[cores,:]
        labels[subj] = tempLabel[cores,:]
        matches[subj] = tempMatch[cores,:]
        
    return [data,labels,matches,coreMaps]

def labelCores(matchingMatrix,threshold):
    
    """
    Method to downsample the trainng data set, based on those vertices that map
    most frequently to any given label.
    
    Originally, we'd thought to threshold the matchingMatrix itself, and choose
    only those vertices with maximum frequencies above a given threshold. This 
    resulted in a situation where some region classes were not represented in
    the training data at all because their maximum mapping frequencies were
    considerably lower than this threshold.
    
    Parameters:
    - - - - -
        matchingMatrix : matrix containing mapping frequency information
                        for each vertex in a training brain
        threshold : percentage of vertces to keep
    """

    matches = matchingMatrix

    maxF = np.max(matches,axis=1)
    zeroInds = maxF > 0
    maxFLabel = np.argmax(matches,axis=1)+1
    maxFLabel = maxFLabel*zeroInds
    
    N = 180
    labels = np.arange(1,N+1)
    
    highestMaps = {}.fromkeys(labels)
    
    for L in labels:
        
        indices = np.where(maxFLabel == L)[0]
        maxFL = maxF[indices]
        
        sortedCoords = np.flip(np.argsort(maxFL),axis=0)
        sortedInds = indices[sortedCoords]
        
        upper = int(np.ceil(threshold*len(sortedInds)))
        
        acceptedInds = sorted(sortedInds[0:upper])
        highestMaps[L] = list(acceptedInds)
    
    return highestMaps