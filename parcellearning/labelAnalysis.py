#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:27:39 2018

@author: kristianeschenburg
"""

import sys
sys.path.insert(0,'../../shortestpath/shortestpath/')
sys.path.insert(1,'../../io/')

import Adjacency as aj
import loaded as ld

import networkx as nx
import numpy as np

def labelErrorDistances(surfaceAdjFile,trueFile,midlineFile,predictedFile,maxValue):
    
    """
    Compute how far away the misclassified label is from the predicted label.
    Distance is based on the geodesic distance map of labels on the cortical
    map.
    
    Parameters:
    - - - - -
        surfaceAdj : surface adjacency file of vertices
        trueFile : true cortical parcellation map file
        midlineFile : midline index file
        predictedFile : predicted cortical parcellation map file
    """
    
    reIndexed = np.zeros((maxValue+1,maxValue+1))

    pred = ld.loadGii(predictedFile,0).astype(np.int32)
    true = ld.loadGii(trueFile,0).astype(np.int32)
    errors = np.where(true != pred)[0]
    
    print len(errors)
    
    L = aj.LabelAdjacency(trueFile,surfaceAdjFile,midlineFile)
    L.generate_adjList()
    
    labelAdj = L.adj_list
    
    # create graph of label adjacency
    GL = nx.from_dict_of_lists(labelAdj)
    
    # get matrix of pairwise shortest paths between all labels
    apsp = nx.floyd_warshall_numpy(GL)
    res_apsp = np.asarray(np.reshape(apsp,np.product(apsp.shape)).T)
    unraveled = np.unravel_index(np.arange(len(res_apsp)),apsp.shape)
    
    newInds = (np.asarray(GL.nodes())[unraveled[0]],
               np.asarray(GL.nodes())[unraveled[1]])
    
    nans = list(set(np.arange(maxValue+1)).difference(set(GL.nodes())))
    
    reIndexed[newInds] = apsp[unraveled]
    reIndexed[nans,:] = np.nan
    reIndexed[:,nans] = np.nan
    
    errorDistance = reIndexed[true[errors],pred[errors]]

    return errorDistance