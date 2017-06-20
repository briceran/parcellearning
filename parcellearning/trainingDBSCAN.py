#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:33:55 2017

@author: kristianeschenburg
"""

import numpy as np
from sklearn import cluster,metrics

import copy

def trainDBSCAN(labelData, eps=0.5, mxs = 10000, mxp = 0.7):
    
    """
    Method to perform DBSCAN for training data.
    
    Paramters:
    - - - - -
        labelData : (dict) training data, keys are labels, values are arrays
        eps : DBSCAN parameter, specifiying maximum distance between two 
                samples for them to be considered as in the same neighborhood
        mxs : maximum number of samples per iteration of DBSCAN
        mxp : minimum percentage of original data points required for a
                completed round of DBSCAN
    """
    
    labels = labelData.keys()
    dbsData = {}.fromkeys(labels)
    
    for lab in labels:
        
        dbsData[lab] = labelDBSCAN(labelData[lab], eps, mxs, mxp)
        
    return dbsData
    
def labelDBSCAN(labelData,eps,max_samples,max_percent):
    
    """
    Method to perform DBSCAN for training data belong to a single label.
    """

    # Shuffle compiled training data for current label.  Shuffling is performed
    # because training data is stacked 1 subject at a time -- we want DBSCAN to
    # find central samples across all training data, not within each subject.
    np.random.shuffle(labelData)
    samples,_ = labelData.shape    

    # if labelData has fewer samples than max_samples, convert to list
    if samples <= max_samples:
        subsets = list([labelData])
        
    # otherwise break into subsets of size max_samples
    # will generally produce one smaller subset
    else:
        iters = samples/max_samples
        subsets = []
        
        for i in np.arange(iters):
            
            bc = i*max_samples
            uc = (i+1)*max_samples
            subsets.append(labelData[bc:uc,:])
        
        subsets.append(labelData[(i+1)*max_samples:,:])

    accepted = []
    
    # for each subset
    for dataSubset in subsets:

        # compute correlation distance (1-corrcoef) and scale to 0-1
        dMat = metrics.pairwise.pairwise_distances(dataSubset,
                                                   metric='correlation')
        dMat = (1-dMat)/2

        perc = 0.0
        ep = copy.copy(eps)

        # while percentage of non-noise samples < max_percentage
        while perc < max_percent:

            # apply DBSCAN, update epsilon parameter (neighborhood size)
            model = cluster.DBSCAN(eps=ep,metric='precomputed',n_jobs=-1)
            model.fit(dMat)
            predLabs = model.labels_            
            clusters = np.where(predLabs != -1)[0]
            
            perc = (1.*len(clusters))/len(predLabs)
            ep += 0.025

        accepted.append(dataSubset[clusters,:])
    
    accepted = np.row_stack(accepted)
    
    return accepted
        
        
