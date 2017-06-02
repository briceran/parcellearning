#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:08:13 2017

@author: kristianeschenburg
"""

from parcellearning import loaded as ld

import os
import nibabel as nb
import numpy as np

def computeAverageShortestPath(dijkstraLabels,predLabels,groundTruth,
                               funcFile,outComp,outFunc):
    
    """
    Method to compute the distance between the ground truth label and the 
    predicted label of every vertex.
    
    Generates a "true distance map" and an "average distance map".  The true
    distance map maintains the true distance for ever very.  The average
    distance map applies the mean distance to every vertex in the truth map.
    
    """
    
    assert os.path.isfile(dijkstraLabels)
    assert os.path.isfile(predLabels)
    assert os.path.isfile(groundTruth)
    assert os.path.isfile(funcFile)
    
    distances = ld.loadPick(dijkstraLabels)
    
    predicted = ld.loadGii(predLabels,0)
    print('pred shape: ',predicted.shape)
    
    truth = ld.loadGii(groundTruth,0)
    print('truth shape: ',truth.shape)
    
    func = nb.load(funcFile)
    
    computed = np.zeros((predicted.shape))
    meaned = np.zeros((predicted.shape))
    
    for i,k in enumerate(truth):

        lab = truth[i]        
        predLab = predicted[i]
        
        if lab != 0 and predLab != 0:
            if distances.has_key(lab):
                if distances[lab].has_key(predLab):

                    computed[i] = distances[lab][predLab]
        else:
            computed[i] = 0
            
    for k in distances.keys():
        
        inds = np.where(truth == k)
        meaned[inds] = np.mean(computed[inds])
        
    func.darrays[0].data = computed.astype(np.float32)
    nb.save(func,outComp)
    
    func.darrays[0].data = meaned.astype(np.float32)
    nb.save(func,outFunc)