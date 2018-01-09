#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:56:48 2017

@author: kristianeschenburg
"""

import copy
import numpy as np

"""
SOFT-MAX THRESHOLDING
"""

def baseSoftMax(metaEstimator,members,memberData,mm,R):
    
    """
    """

    print 'Base Prediction'
    
    predicted = np.squeeze(metaEstimator.predict(memberData))

    return predicted


def forestSoftMax(metaEstimator,members,memberData,mm,R):
    
    """
    Method to restrict whole random decision forest soft-max prediction to
    labels generated in the surface registration step.
    
    Parameters:
    - - - - -
        
        metaEstimator : single decision forest
                
        members : current vertices of interest
        
        memberData : feature data for members
        
        mm : binary matching matrix
        
        R : number of labels in training set
    """

    memberMatrix = mm[members,:]
    predProbs = metaEstimator.predict_proba(memberData)
    
    classes = metaEstimator.classes_
    forestMatrix = np.zeros((len(members),mm.shape[1]+1))
    forestMatrix[:,classes] = predProbs
    forestMatrix = forestMatrix[:,1:]
    
    predThresh = memberMatrix*(1.*forestMatrix)
    
    labels = np.argmax(predThresh,axis=1)+1

    return labels


def treeSoftMax(metaEstimator,members,memberData,mm,R):
    
    """
    Super method for classification prediction with tree-level restriction.
    
    Parameters:
    - - - - -
        metaEstimator : single decision forest
                
        members : current vertices of interest
        
        memberData : feature data for members
        
        mm : binary matching matrix
        
        R : number of labels in training set
        
    A single metaEstimator consists of a set of sub-estimators.  The classes 
    for the metaEstimator corresponds to a list of K values to predict 
    i.e. [1,5,6,7,100]
    
    Each sub-estimator, however, has 0-K indexed classes (i.e. [0,1,2,3,4]), 
    where the index corresponds to the position in the metaEstimator 
    class list.
    """

    print 'Tree Prediction'

    memberMatrix = mm[members,:]

    treeProbs = []
    classes = []
    for est in metaEstimator.estimators_:
        classes.append(metaEstimator.classes_)
        treeProbs.append(est.predict_proba(memberData))

    predictedLabels = []
    
    treeMatrix = np.zeros((len(members),mm.shape[1]+1))
        
    for k in np.arange(len(treeProbs)):
        
        treeClasses = np.squeeze(classes[k]).astype(np.int32)
        treeProbabilities = treeProbs[k]
        
        tm = copy.deepcopy(treeMatrix)
        tm[:,treeClasses] = treeProbabilities
        tm = tm[:,1:]
        
        treeThresh = memberMatrix*(1.*tm)
        predictedLabels.append(np.argmax(treeThresh,axis=1)+1)
    
    predictedLabels = np.column_stack(predictedLabels)

    classification = []
        
    for i in np.arange(predictedLabels.shape[0]):
        
        L = list(predictedLabels[i,:])
        maxProb = max(set(L),key=L.count)
        classification.append(maxProb)
    
    classification = np.asarray(classification)
        
    return classification