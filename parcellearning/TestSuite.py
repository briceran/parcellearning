#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:57:32 2017

@author: kristianeschenburg
"""

"""
Test suite defines a set of methods to assess the performance of a classifier.
These methods operate on a per-label basis to generate an H5 file for each
predicted label.  The metrics can be accessed in a similar manner to 
accessing data in a dictionary -- that is, with key-value pairs.
"""

import numpy as np
import nibabel as nb
import copy

def atlasOverlap(atlasMap,cbpLabel,A,L):
    
    """
    Compute the overlap of the predicted label file with the label file of a
    given atlas.  For example, we might want to compute the overlap of the
    connectivity-based map with the Destrieux atlas, or Desikan-Killiany atlas.
    
    Parameters:
    - - - - -
        atlasMap : dictionary, where the key is the name of the atlas map 
                    i.e. 'Destrieux', and the value is the file path
        
        cbpLabel : file path to the predicted label
        
        A : number of expected regions in atlas map
        
        L : number of expected regions in the predicted map
    """
    
    atlName = atlasMap['name']
    atlFile = atlasMap['file']
    
    atl = nb.load(atlFile)
    atl = atl.darrays[0].data
    atlLabels = list(set(atl).difference({0}))
    print atlLabels
    
    cbp = nb.load(cbpLabel)
    cbp = cbp.darrays[0].data
    cbpLabels = list(set(cbp).difference({0}))
    
    overlaps = np.zeros((L+1,A+1))
    
    cbpIndices = {}.fromkeys(np.arange(1,L))
    atlIndices = {}.fromkeys(np.arange(1,A+1))
    
    for c in cbpLabels:
        cbpIndices[c] = np.where(cbp == c)[0]
    
    for a in atlLabels:
        atlIndices[a] = np.where(atl == a)[0]
    
    print 'Entering loop'
    for c in cbpLabels:
        cbpInds = cbpIndices[c]
        
        for a in atlLabels:
            atlInds = atlIndices[a]
            
            if len(atlInds) and len(cbpInds):
                
                ov = len(set(cbpIndices[c]).intersection(set(atlIndices[a])))
                overlaps[c,a] = (1.*ov)/len(cbpIndices[c])
            else:
                overlaps[c,a] = 0
    
    return [atlName,overlaps]

def accuracy(cbpLabel,trueLabel):
    
    """
    Compute the classification accuracy of the predicted label as compared to
    the ground truth label.
    """
    
    cbp = np.load(cbpLabel)
    cbp = cbp.darrays[0].data
    
    truth = nb.load(trueLabel)
    truth = truth.darrays[0].data
    
    return np.mean(cbp == truth)

def modelHomogeneity(data,truthLabel,cbpLabel,L,iters):
    
    """
    Compute homogeneity for a predicted label, its truth label, and random
    permutations of the predicted label.
    
    Parameters:
    - - - - -
        truthLabel : true cortical map
        
        cbpLabel : predicted cortical map
        
        L : expected number of regions
        
        iters : number of random permutations
    """
    
    truth = copy.deepcopy(truthLabel)
    cbp = copy.deepcopy(cbpLabel)

    hmg = {}.fromkeys(['truth','predicted','random'])
    hmg['random'] = {}.fromkeys(np.arange(1,L+1))

    hmg['truth'] = homogeneity(truth,data,180)
    hmg['predicted'] = homogeneity(cbpLabel,data,180)
    
    permutes = np.zeros((L+1,iters))
    
    for k in np.arange(0,iters):
        
        cbp = copy.deepcopy(cbp)

        np.random.shuffle(cbp)
        p = homogeneity(cbp,data,L)
        permutes[p.keys(),k] = p.values()
        print permutes[:,1] == permutes[:,2]

    for lab in np.arange(1,L+1):
        hmg['random'][lab] = permutes[lab,:]
    
    return (hmg,permutes)

def homogeneity(labelArray,trainingData,L):
    
    """
    Mean the homogeneity of each region with regard to its feature vectors.
    
    Parameters:
    - - - - -
        labelArray : vector of labels
        
        L : expected number of labels
    """
    
    regional = {}.fromkeys(np.arange(1,L+1))
    
    for lab in regional.keys():
        inds = np.where(labelArray == lab)[0]
        if len(inds) > 1:
            data = trainingData[inds,:]
            sims = np.corrcoef(data)
            regional[lab] = np.mean(sims)
        elif len(inds) == 1:
            regional[lab] = 1
        else:
            regional[lab] = 0;
            
    return regional

