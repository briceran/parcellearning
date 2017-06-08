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
            
            # if single elements, convert to lists
            if isinstance(sg_border,int):
                sg_border = list([sg_border])
            
            if isinstance(sg_internal,int):
                sg_internal = list([sg_internal])
                
            for i,n in enumerate(sg_internal):
                # iterate over border vertices
                for b in sg_border:
                    if nx.has_path(subGraph,source=n,target=b):
                        sg_nb = nx.shortest_path_length(subGraph,source=n,
                                                        target=b)
                    else:
                        sg_nb = None
                    distances[n].append(sg_nb)
    
    for key in distances.keys():
        if distances[key]:
            if isinstance(distances[key],list):
                distances[key] = min(distances[key])
        else:
            distances[key] = None

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
    """
    
    # load files
    labAdj = ld.loadPick(labelAdjacency)
    truth = ld.loadGii(truthLabFile,0)
    pred = ld.loadGii(predLabFile,0)
    
    # initialize new color map file
    color_file = open(outputColorMap,"w")
    
    # extract current colors from colormap
    parsedColors = parseColorLookUpFile(labelLookup)
    
    # get labels that neighbor core label
    neighbors = labAdj[core]
    # get indices of core label in true map
    truthInds = np.where(truth == core)[0]
    
    # initialize new map
    visualizeMap = np.zeros((truth.shape))
    visualizeMap[truthInds] = core
    
    trueColors = [255,255,255]
    trueColors = map(str,trueColors)
    
    trueName = 'Label {}'.format(core)
    trueRGBA = '{} {} {}\n'.format(core,' '.join(trueColors),255)
    
    trueStr = '\n'.join([trueName,trueRGBA])
    color_file.writelines(trueStr)
    
    # get predicted label values existing at truthInds
    predLabelsTruth = pred[truthInds]
    print('confused labels: ',set(predLabelsTruth))

    for n in neighbors:
        
        originalLabel = n
        
        # get color code for label, and adjust
        colorCode = parsedColors[n]
        adjColors = shiftColor(colorCode,mag=20)
        adjColors = map(str,adjColors)
        
        # convert original code to string
        originalColorStr = map(str,colorCode)
        
        n_inds = np.where(truth == originalLabel)[0]
        visualizeMap[n_inds] = originalLabel
        
        # write original label data to text file
        originalLabelName = 'Label {}'.format(originalLabel)
        originalRGBA = '{} {} {}\n'.format(originalLabel,
                        ' '.join(originalColorStr),255)
        originalStr = '\n'.join([originalLabelName,originalRGBA])
        
        color_file.writelines(originalStr)
        
        # create new label, outside original bounds
        newLabel = originalLabel+180
        
        # write new label data to text file
        label_name = 'Label {}'.format(newLabel)
        labelRGBA  = '{} {} {}\n'.format(newLabel,' '.join(adjColors),255)
        newStr = '\n'.join([label_name,labelRGBA])
        
        color_file.writelines(newStr)
        
        n_inds = np.where(predLabelsTruth == n)[0]
        visualizeMap[truthInds[n_inds]] = newLabel
    
    color_file.close()
    
    
    
    return visualizeMap
        
        
        
        