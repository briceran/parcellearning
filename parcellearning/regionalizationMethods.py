#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:08:50 2017

@author: kristianeschenburg
"""

import loaded as ld
import numpy as np
import networkx as nx

from sklearn import metrics

from joblib import Parallel, delayed
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

#####
"""
Methods relating to performing regionalization of a time series, using the 
data provided in the level structures.
"""
#####

def regionalizeStructures(timeSeries,levelStructures,level,midlines,
                          measure='median',R=180):
    
    """
    Method to regionalize the resting state connectivity, using only vertices
    included at a minimum level away from the border vertices.
    
    Parameters:
    - - - - -
        timeSeries : input resting state file
        levelStrucutres : levelStructures file created by computeLabelLayers
        level : depth to constaint layers at
        midlines : path to midline indices
        measur
    """
    
    assert measure in ['median','mean']
    assert level >= 1
    
    
    resting = ld.loadMat(timeSeries)
    midlines = ld.loadMat(midlines)
    levelSets = ld.loadPick(levelStructures)
    
    resting[midlines,:] = 0
    
    condensedLevels = layerCondensation(levelSets,level)
    
    regionalized = np.zeros((resting.shape[0],R))
    
    for region_id in condensedLevels.keys():
        
        print(region_id)
        
        subregion = condensedLevels[region_id]
        if len(subregion):
            subregion = list(set(subregion).difference(set(midlines)))
            
            subrest = resting[subregion,:]
            
            correlated = metrics.pairwise.pairwise_distances(resting,subrest,
                                                             metric='correlation')

            if measure == 'median':
                regionalized[:,region_id-1] = np.median(1-correlated,axis=1)
            else:
                regionalized[:,region_id-1] = np.mean(1-correlated,axis=1)
        
    return regionalized
        
        
        
        
        
        

#####
"""
Methods to compute level structures on a cortical map file
"""
#####

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
            neighborLabels = list(label[neighbors])
            
            # if vertex is isolated instance of lab (i.e. noise), exclude it
            selfLabels = neighborLabels.count(lab)
            
            if len(set(neighborLabels)) > 1 and selfLabels > 0:
                
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
    
    # get set of non-zero labels in label file
    L = set(label) - set([0])
    
    layers = {}.fromkeys(L)
    
    fullList = Parallel(n_jobs=NUM_CORES)(delayed(labelLayers)(lab,
                        np.where(label == lab)[0],
                        surfAdj,borders[lab]) for lab in L)
    
    for i,lab in enumerate(L):
        layers[lab] = fullList[i]

    return layers

def labelLayers(lab,labelIndices,surfAdj,borderIndices):
    
    """
    Method to compute level structures for a single region.
    
    Parameters:
    - - - - -
        labelIndices : indices of whole ROI
        surfAdj : surface adjacency file corresponding to whole surface
        borderIndices : indices corresponding to border of ROI
    """
    
    print ('Computing layers for label {}.'.format(lab))

    internalNodes = list(set(labelIndices).difference(borderIndices))
    
    # compute condensed adjacency list corresponding to vertices in ROI
    regionSurfAdj = {k: [] for k in labelIndices}
    
    for li in labelIndices:
        
        fullNeighbs = surfAdj[li]
        regionSurfAdj[li] = list(set(labelIndices).intersection(fullNeighbs))
        
    # generate graph of condensed surface adjacency file
    G = nx.from_dict_of_lists(regionSurfAdj)
    
    distances = {n: [] for n in internalNodes}
    
    # here, we allow for connected components in the regions
    for subGraph in nx.connected_component_subgraphs(G):
        
        # get subgraph nodes
        sg_nodes = subGraph.nodes()
        
        # make sure subgraph has more than a single component
        if len(sg_nodes) > 1:
            
            # get subgraph border indices
            sg_border = list(set(sg_nodes).intersection(borderIndices))
            # get subgraph internal indices
            sg_internal = list(set(sg_nodes).intersection(internalNodes))
            
            sp = nx.all_pairs_shortest_path_length(subGraph)
            
            for k in sg_internal:
                distances[k] = [v for j,v in sp[k].items() if j in sg_border]
                distances[k] = min(distances[k])

    layered = {k: [] for k in set(distances.values())}
    
    for vertex in distances.keys():
        dist = distances[vertex]
        layered[dist].append(vertex)
        
    return layered      

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
        
        if len(deepVertices):
            
            deepVertices = np.concatenate(deepVertices)
            condensedLayers[k] = deepVertices
        
    return condensedLayers

#####
"""
Methods relating to visualizing predicted cortical maps.
"""
#####

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

def shiftColor(rgb,mag=30):
    
    """
    Method to adjust rgba slightly slightly.
    """
    
    rgb_adj = []
    
    for i in np.arange(3):
        
        r = np.random.choice([-1,1])
        m = (1*r*mag)
        
        adj = rgb[i]+m
        
        if adj > 255:
            adj = adj - 2*mag
        elif adj < 1:
            adj = adj + 2*mag
        
        rgb_adj.append(adj)
    
    return rgb_adj

def neighborhoodErrorMap(core,labelAdjacency,truthLabFile,
                         predLabFile,labelLookup,outputColorMap):
    
    """
    Method to visualize the results of a prediction map, focusing in on a 
    spefic core label.
    
    Parameters:
    - - - - -
        core : region of interest
        labelAdjacency : label adjacency list
        truthLabFile : ground truth label file
        predLabFile : predicted label file
        labelLookup : label color lookup table
        outputColorMap : new color map for label files
    """
    
    # load files
    labAdj = ld.loadPick(labelAdjacency)
    truth = ld.loadGii(truthLabFile,0)
    pred = ld.loadGii(predLabFile,0)
    
    # extract current colors from colormap
    parsedColors = parseColorLookUpFile(labelLookup)
    


    # initialize new color map file
    color_file = open(outputColorMap,"w")

    trueColors = ' '.join(map(str,[255,255,255]))
    trueName = 'Label {}'.format(core)
    trueRGBA = '{} {} {}\n'.format(core,trueColors,255)
    
    trueStr = '\n'.join([trueName,trueRGBA])
    color_file.writelines(trueStr)
    
    
    
    
    # get labels that neighbor core
    neighbors = labAdj[core]
    # get indices of core label in true map
    truthInds = np.where(truth == core)[0]
    
    # initialize new map
    visualizeMap = np.zeros((truth.shape))
    visualizeMap[truthInds] = core
    
    # get predicted label values existing at truthInds
    predLabelsTruth = pred[truthInds]

    for n in neighbors:
        
        # get color code for label, adjust and write text to file
        oriName = 'Label {}'.format(n)
        oriCode = parsedColors[n]
        oriColors = ' '.join(map(str,oriCode))
        oriRGBA = '{} {} {}\n'.format(n,oriColors,255)
        oriStr = '\n'.join([oriName,oriRGBA])
        
        color_file.writelines(oriStr)
        
        adjLabel = n+180
        adjName = 'Label {}'.format(adjLabel)
        adjColors = shiftColor(oriCode,mag=20)
        adjColors = ' '.join(map(str,adjColors))
        adjRGBA = '{} {} {}\n'.format(adjLabel,adjColors,255)
        adjStr = '\n'.join([adjName,adjRGBA])
        
        color_file.writelines(adjStr)
        
        
        # find where true map == n and set this value
        n_inds = np.where(truth == n)[0]
        visualizeMap[n_inds] = n
        
        # find where prediction(core) == n, and set to adjusted value
        n_inds = np.where(predLabelsTruth == n)[0]
        visualizeMap[truthInds[n_inds]] = adjLabel
    
    color_file.close()

    return visualizeMap
        
        
        
        