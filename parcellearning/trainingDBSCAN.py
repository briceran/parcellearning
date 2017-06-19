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
        
        dbsData[lab] = labelDBSCAN(lab, labelData[lab], eps, mxs, mxp)
        
    return dbsData
    

def labelDBSCAN(label,labelData,eps,max_samples,max_percent):
    
    """
    Method to perform DBSCAN for training data belong to a single label.
    """
    
    print 'input shape: {}'.format(labelData.shape)
    
    np.random.shuffle(labelData)
    samples,_ = labelData.shape    

    if samples <= max_samples:
        subsets = list([labelData])
        
    else:
        iters = samples/max_samples
        subsets = []
        
        for i in np.arange(iters):
            
            bc = i*MAXSAMPLES
            uc = (i+1)*MAXSAMPLES
            subsets.append(labelData[bc:uc,:])
        
        subsets.append(labelData[(i+1)*MAXSAMPLES:,:])

    accepted = []
    
    for dataSubset in subsets:

        ccoef = np.corrcoef(dataSubset)
        dMat = (1-ccoef)/2

        perc = 0.0
        ep = copy.copy(eps)

        while perc < max_percent:

            model = cluster.DBSCAN(eps=ep,metric='precomputed',n_jobs=-1)
            model.fit(dMat)
            predLabs = model.labels_            
            clusters = np.where(predLabs != -1)[0]
            
            perc = (1.*len(clusters))/len(predLabs)
            ep += 0.025

        tempAcc = dataSubset[clusters,:]
        accepted.append(tempAcc)
    
    accepted = np.row_stack(accepted)
    
    return accepted
        
        
