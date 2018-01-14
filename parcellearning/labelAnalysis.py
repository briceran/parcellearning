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

def labelErrorDistances(surfaceAdjFile,trueFile,midlineFile,predictedFile):
    
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
    
    
    pred = ld.loadGii(predictedFile,0)
    true = ld.loadGii(trueFile,0)
    errors = np.where(true != pred)[0]
    errorDistance = np.zeros((len(errors),))
    
    L = aj.LabelAdjacency(trueFile,surfaceAdjFile,midlineFile)
    L.generate_adjList()
    
    labelAdj = L.adj_list
    
    GL = nx.from_dict_of_lists(labelAdj)
    apsp = nx.all_pairs_shortest_path_length(GL)
        
    for j,e in enumerate(errors):
        trueLabel = true[e]
        predLabel = pred[e]

        if predLabel in set(true):
            errorDistance[j] = apsp[trueLabel][int(predLabel)]
        else:
            errorDistance[j] = np.nan
    
    return errorDistance