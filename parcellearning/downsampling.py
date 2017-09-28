#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:07:37 2017

@author: kristianeschenburg
"""

import copy
import numpy as np
import sys

import dataUtilities as du

def byCore(data,response,matches,labels,fraction=0.7):
    
    """
    Downsample the training data for each subject, for each label, by
    selecting the core vertices for each label.

    Parameters:
    - - - - -
        data : dictionary of training data, where keys are subjects
                        and values are the vertex-wise data arrays
        response : dictionary of training labels, where keys are subjects
                            and values are vertex-wise label assignments
        matches : dictionary of matches, where keys are subjects, and
                            values are vertex-to-label frequency arrays
        labels : set of unique labels across all training subjects
    Returns:
    - - - -
        x : downsampled data array dictionaries
        y : downsampled response vector dictionaries
        m : downsampled match array dictionaries
        coreMaps : indices of "good" vertices, per subject, per label
    """
    
    assert data.keys() == response.keys()
    
    x = copy.deepcopy(data)
    y = copy.deepcopy(response)
    m = copy.deepcopy(matches)
    
    subjects = x.keys()
    coreMaps = {}.fromkeys(subjects)

    for subj in x.keys():
        
        tempData = x[subj]
        tempLabel = y[subj]
        tempMatch = m[subj]

        cores = getCores(tempMatch,fraction)
        coreMaps[subj] = cores
        
        cores = np.squeeze(np.concatenate(cores.values())).astype(np.int32)
        cores = np.sort(cores)

        x[subj] = tempData[cores,:]
        y[subj] = tempLabel[cores,:]
        m[subj] = tempMatch[cores,:]
        
    return [x,y,m]


def getCores(matches,fraction):
    
    """
    Computes core vertices for each label, based on the frequency with which
    vertices map to a label.
    
    Parameters:
    - - - - -
        matchingMatrix : matrix containing mapping frequency information
                        for each vertex in a training brain
        fraction : fraction of vertces to keep
    """
    
    # compute max mapping frequency for each vertex
    maxF = np.max(matches,axis=1)
    # find vertices with only frequencies
    posInds = maxF > 0
    
    # compute maximum frequency labels
    maxFLabel = np.argmax(matches,axis=1)+1
    # mask labels by pos frequencies
    maxFLabel = maxFLabel*posInds
    
    N = matches.shape[1]-1
    labels = np.arange(1,N+1)
    
    highestMaps = {}.fromkeys(labels)
    
    for lab in labels:
        
        # get indices of vertex with maxiumum frequency label == L
        indices = np.where(maxFLabel == lab)[0]
        maxFL = maxF[indices]
        
        # sort these vertices' frequencies from high to low
        sortedCoords = np.flip(np.argsort(maxFL),axis=0)
        sortedInds = indices[sortedCoords]
        
        # select highest fraction of these
        upper = int(np.ceil(fraction*len(sortedInds)))
        
        # sort and return indices
        acceptedInds = sorted(sortedInds[0:upper])
        highestMaps[lab] = list(acceptedInds)
    
    return highestMaps


def byMinimum(data,response,matches,labels):
    
    """
    Downsamples the training data to match size of smallest-sample label.
    
    Parameters:
    - - - - -
        data : dictionary of training data, where keys are subjects
                        and values are the vertex-wise data arrays
        labels : dictionary of training labels, where keys are subjects
                            and values are vertex-wise label assignments
        matches : dictionary of matches, where keys are subjects, and
                            values are vertex-to-label frequency arrays
        labels : set of unique labels across all training subjects
    Returns:
    - - - -
        pData : downsampled data array
        pLabels : downsampled response vectors
        pMatches : downsampled matches
    """
    
    data = du.mergeValueArrays(data)
    response = du.mergeValueLists(response)
    matches = du.mergeValueArrays(matches)
    
    minSize = sys.maxint
    
    pData = du.splitArrayByResponse(data,response,labels)
    pMatches = du.splitArrayByResponse(matches,response,labels)
    
    pLabels = du.buildResponseVector(pData)
    
    # compute minimum size sample array
    for lab in labels:
        
        tempData = pData[lab]
        minSize = min(minSize,tempData.shape[0])
    
    # downsample remaining arrays
    for lab in labels:
        
        tempData = pData[lab]
        tempMatches = pMatches[lab]
        tempLabels = pLabels[lab]
        
        inds = np.random.choice(np.arange(tempData.shape[0],size=minSize,
                                          replace=False))
        pData[lab] = tempData[inds,:]
        pMatches[lab] = tempMatches[inds,:]
        pLabels[lab] = tempLabels[inds,:]
        
    return [pData,pLabels,pMatches]