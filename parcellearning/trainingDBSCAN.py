#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:33:55 2017

@author: kristianeschenburg
"""

import numpy as np
from sklearn import cluster,metrics

import copy

MAXSAMPLES = 7500
MAXPERCENT = 0.7

def trainDBSCAN(labelData,eps=0.5):
    
    """
    Method to perform DBSCAN for training data.
    
    Paramters:
    - - - - -
        labelData : (dict) training data, keys are labels, values are arrays
        eps : DBSCAN parameter, specifiying maximum distance between two 
                samples for them to be considered as in the same neighborhood
        metric : metric to compute distance matrix
    """
    
    labels = labelData.keys()
    dbsData = {}.fromkeys(labels)
    
    for lab in labels:
        
        dbsData[lab] = labelDBSCAN(lab,labelData[lab],eps)
        
    return dbsData
    

def labelDBSCAN(label,labelData,eps):
    
    """
    Method to perform DBSCAN for training data belong to a single label.
    """
    
    print 'input shape: {}'.format(labelData.shape)
    
    np.random.shuffle(labelData)
    samples,_ = labelData.shape    

    if samples <= MAXSAMPLES:
        subsets = list([labelData])
        
    else:
        iters = samples/MAXSAMPLES
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

        while perc < MAXPERCENT:

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
        
        
