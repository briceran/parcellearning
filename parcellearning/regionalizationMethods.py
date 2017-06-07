#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:08:50 2017

@author: kristianeschenburg
"""

import loaded as ld
import numpy as np
import networkx as nx

from joblib import Parallel, delayed
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

def coreBoundaryVertices(labelFile,surfaceAdjacency):
    
    """
    Method to find the border vertices of each label.  These will be stored
    in a dictionary, where keys are labels and values are the boundary indices.

    Parameters:
    - - - - -
        labelFile : cortical map file
        surfaceAdjacency : surface adjacency file
    """
    
    label = ld.loadGii(labelFile,0)
    surfAdj = ld.loadPick(surfaceAdjacency)

    L = set(label) - set([0])
    
    borderVertices = {l: [] for l in L}
    
    for lab in L:
        
        inds = np.where(label == lab)[0]
        
        for i in inds:
            
            # get neighboring vertices of vertex i
            neighbors = surfAdj[i]
            # get labels of neighbors of vertex i
            neighborLabels = label[neighbors]
            
            # if vertex is isolated instance of lab (i.e. noise), exclude it
            selfLabels = neighborLabels.count(lab)
            
            if len(set(neighborLabels)) > 1 and len(selfLabels) > 1:
                
                borderVertices[lab].append(i)
    
    return borderVertices

def computeLabelLayers(labelFile,surfaceAdjacency,borderFile):
    
    """
    Method to find level structures of vertices, where each structure is a set
    of vertices that are a distance k away from the border vertices.
    """
    
    label = ld.loadGii(labelFile,0)
    surfAdj = ld.loadPick(surfaceAdjacency)
    borders = ld.loadPick(borderFile)
    
    L = set(label) - set([0])
    
    Layers = {}.fromkeys(L)
    
    layers = Parallel(n_jobs=NUM_CORES)(delayed(labelLayers)(lab,
                      np.where(label == lab)[0],
                      surfAdj,borders[lab]) for lab in L)
    
    for i,l in enumerate(L):

        Layers[l] = layers[i]
        
    return Layers

def labelLayers(lab,labelIndices,surfAdj,borderIndices):
    
    """
    Method to compute level structures for a single region.
    
    Parameters:
    - - - - -
        labelIndices : indices of whole ROI
        surfAdj : surface adjacency file corresponding to whole surface
        borderIndices : indices corresponding to border of ROI
    """
    
    print('Computing layers for label {}.'.format(lab))
    
    regionSurfAdj = {k: [] for k in labelIndices}
    
    # compute condensed adjacency list corresponding to vertices in ROI
    for li in labelIndices:
        
        fullNeighbs = surfAdj[li]
        regionSurfAdj[li] = list(set(labelIndices).intersection(fullNeighbs))
        
    G = nx.from_dict_of_lists(regionSurfAdj)
        
    nonBorders = list(set(labelIndices).difference(borderIndices))
    
    distances = {n: [] for n in nonBorders}
    
    for i,n in enumerate(nonBorders):
        for b in borderIndices:
            # see if there is a path between the source node and target node
            try:
                sp_nb = nx.shortest_path_length(G,source=n,target=b)
            # otherwise set distance to none
            except:
                sp_nb = None

            distances[n].append(sp_nb)
        
        distances[n] = min(distances[n])
        
    layers = {k: [] for k in distances.values()}
    
    for vertex in distances.keys():
        dist = distances[vertex]
        layers[dist].append(vertex)
        
    return layers      

def layerCondensation(layers,level):
    
    """
    Method to condense vertices of layers at at least a depth of level.
    
    Parameters:
    - - - - -
        layers : layers for each label
        level : minimum distance a vertex needs to be from the boundary
                vertices.
    """
    
    condensedLayers = {k: [] for k in layers.keys()}
    
    for k in layers.keys():
        
        k_label = layers[k]
        
        deepVertices = [v for j,v in k_label.items() if j >= level]
        deepVertices = np.concatenate(deepVertices)
        
        condensedLayers[k] = deepVertices
        
    return condensedLayers

def parseColorLookUpFile(lookupTable):
    
    """
    Method to convert 
    """
    
    with open(lookupTable,"rb") as input:
        lines = input.readlines()
    
    lines = [map(int,v.strip().split(' ')) for i,v in enumerate(lines) if i % 2 == 1]
    
    lines = np.row_stack(lines)
    
    parsedColors = {k: list(v) for k,v in zip(lines[:,0],lines[:,1:4])}
    
    return parsedColors

def shiftColor(rgb,mag=20):
    
    """
    Method to adjust rgba slightly slightly.
    """
    
    rgb_adj = []
    
    for i in np.arange(3):
        
        r = np.random.choice([-1,1])
        print(r)
        m = (1*r*mag)
        
        adj = rgb[i]+m
        
        if adj > 255:
            adj = adj - 2*mag
        elif adj < 1:
            adj = adj + 2*mag
        
        rgb_adj.append(adj)
    
    return rgb_adj
        
        