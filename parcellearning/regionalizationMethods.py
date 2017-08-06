#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:08:50 2017

@author: kristianeschenburg
"""

import loaded as ld

import networkx as nx
import nibabel as nb
import numpy as np
from sklearn import cluster,metrics

import copy
import os

#####
"""
Methods relating to performing regionalization of samples / time series
using the data provided in the level structures.
"""
#####

def regionalizeStructures(timeSeries,levelStructures,midlines,level,R,
                          measure='median'):
    
    """
    Method to regionalize the resting state connectivity, using only vertices
    included at a minimum level away from the border vertices.
    
    Parameters:
    - - - - -
        timeSeries : input resting state file
        levelStrucutres : levelStructures created by computeLabelLayers
                            ".RegionalLayers.p" file
        level : depth to constaint layers at
        midlines : path to midline indices
        measure : measure to apply to correlation values ['mean','median']
    """
    
    assert measure in ['median','mean']
    assert level >= 1
    
    
    resting = ld.loadMat(timeSeries)
    midlines = ld.loadMat(midlines)
    levelSets = ld.loadPick(levelStructures)
    
    resting[midlines,:] = 0
    
    condensedLevels = layerCondensation(levelSets,level)
    
    regionalized = np.zeros((resting.shape[0],R))
    
    # get the central vertices for region_id
    for region_id in condensedLevels.keys():
        
        print(region_id)
        
        subregion = condensedLevels[region_id]
        subregion = list(set(subregion).difference(set(midlines)))
        print('# subvertices: {}'.format(len(subregion)))
        
        # if subregion has at least 1 vertex
        if len(subregion):

            subrest = resting[subregion,:]
            
            if np.ndim(subrest) == 1:
                subrest.shape += (1,)
            
            if subrest.shape[1] != resting.shape[1]:
                subrest = subrest.T

            correlated = metrics.pairwise.pairwise_distances(resting,subrest,
                                                             metric='correlation')

            if measure == 'median':
                regionalized[:,region_id-1] = np.median(1-correlated,axis=1)
            else:
                regionalized[:,region_id-1] = np.mean(1-correlated,axis=1)
    
    regionalized[midlines,:] = 0
        
    return regionalized


def trainDBSCAN(labelData, eps=0.025, mxs = 7500, mxp = 0.7):
    
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
    
    labDat = copy.deepcopy(labelData)
    
    labels = labDat.keys()
    dbscanCoords = {}.fromkeys(labels)

    for lab in labels:
        print 'Label {}'.format(lab)
        tempData = labDat[lab]
        dbscanCoords[lab] = findLabelDBSCoords(lab,tempData,eps,mxs,mxp)
        
    return dbscanCoords
    
def findLabelDBSCoords(label,data,eps,max_samples,max_percent):
    
    """
    Method to perform DBSCAN for training data belong to a single label.
    """

    # Shuffle compiled training data for current label.  Shuffling is performed
    # because training data is stacked 1 subject at a time -- we want DBSCAN to
    # find central samples across all training data, not within each subject.
    np.random.shuffle(data)
    samples,_ = data.shape    

    # if labelData has fewer samples than max_samples, convert to list
    if samples <= max_samples:
        subsets = list([data])
        
    # otherwise break into subsets of size max_samples
    # will generally produce one smaller subset
    else:
        iters = samples/max_samples
        subsets = []
        
        for i in np.arange(iters):
            print 'Iteration {}'.format(i)
            
            bc = i*max_samples
            uc = (i+1)*max_samples
            subsets.append(data[bc:uc,:])
        
        subsets.append(data[(i+1)*max_samples:,:])

    coordinates = []

    # for each subset
    baseline = 0
    for iteration,dataSubset in enumerate(subsets):
        
        [xSamps,yDim] = dataSubset.shape

        # compute correlation distance (1-corrcoef) and scale to 0-1
        dMat = metrics.pairwise.pairwise_distances(dataSubset,
                                                   metric='correlation')
        dMat = dMat/2

        perc = 0.0
        ep = copy.copy(eps)

        # while percentage of non-noise samples < max_percentage
        while perc < max_percent:

            # apply DBSCAN, update epsilon parameter (neighborhood size)
            model = cluster.DBSCAN(eps=ep,metric='precomputed',n_jobs=-1)
            model.fit(dMat)
            predLabs = model.labels_            
            clusters = np.where(predLabs != -1)[0]

            perc = (1.*len(clusters))/(1.*len(predLabs))
            ep += 0.01
        
        clusters += baseline
        baseline += xSamps

        coordinates.append(clusters)

    coordinates = np.concatenate(coordinates)

    return coordinates
        
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
    
    """
    fullList = Parallel(n_jobs=NUM_CORES)(delayed(labelLayers)(lab,
                        np.where(label == lab)[0],
                        surfAdj,borders[lab]) for lab in L)
    """
    
    for i,labelValue in enumerate(L):
        if borders.has_key(labelValue):
        
            inds = np.where(label == labelValue)[0]
            bm = borders[labelValue]
    
            layers[labelValue] = labelLayers(labelValue,inds,surfAdj,bm)

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
            
            sg_border = list(set(sg_nodes).intersection(borderIndices))
            sg_intern = list(set(sg_nodes).intersection(internalNodes))
            
            external = [i for i,j in enumerate(sg_nodes) if j in sg_border]
            
            sp = nx.floyd_warshall_numpy(subGraph)
            se = sp[:,external]
            se = se.astype(np.int32)
            
            for x in np.arange(se.shape[0]):
                print "X: {}".format(x)
                print set(list(np.squeeze(np.asarray(se[x,:]))))

            minDist = np.squeeze(np.min(se,axis=1))
            
            if lab == 131:
                print minDist
                print "min24: {}".format(minDist[24])

            for k,v in enumerate(sg_nodes):
                if v in sg_intern:
                    distances[v] = int(np.min(list(set(list(np.squeeze(np.asarray(se[k,:])))))))
                    #distances[v] = int(np.asarray(minDist[k]))

    print 'Label {} layers'.format(lab)
    D = distances.values()
    md = set()
    for l in D:
        if l!=[]:
            md.add(l)
    
    layered = {k: [] for k in md}
    
    for j,vertex in enumerate(distances.keys()):
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

def neighborhoodErrorMap(labVal,labelAdjacency,truthLabFile,
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
    trueName = 'Label {}'.format(labVal)
    trueRGBA = '{} {} {}\n'.format(labVal,trueColors,255)
    
    trueStr = '\n'.join([trueName,trueRGBA])
    color_file.writelines(trueStr)
    
    
    
    
    # get labels that neighbor core
    neighbors = labAdj[labVal]
    # get indices of core label in true map
    truthInds = np.where(truth == labVal)[0]
    
    # initialize new map
    visualizeMap = np.zeros((truth.shape))
    visualizeMap[truthInds] = labVal
    
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
        adjColors = shiftColor(oriCode,mag=30)
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

def processNeighborhoodCM(labVal,labelAdjacency,truthLabFile,
                         predLabFile,labelLookup,inMyl,outDir):
    
    ocm = outDir + '{}.ColorMap.txt'.format(labVal)
    
    vm = neighborhoodErrorMap(labVal,labelAdjacency,truthLabFile,
                         predLabFile,labelLookup,ocm)
        
    outFunc = outDir + 'Label_{}.LabelMisMatch.func.gii'.format(labVal)
    myl = nb.load(inMyl)
    myl.darrays[0].data = vm.astype(np.float32)
    nb.save(myl,outFunc)
    
    outLabel = outDir + 'Label_{}.LabelMisMatch.label.gii'.format(labVal)

    #cmd_call = ['wb_command','-metric-label-import',outFunc,ocm,outLabel]
    cmd_call = 'wb_command -metric-label-import {} {} {}'.format(outFunc,ocm,outLabel)
    
    print(os.path.isfile(outFunc))
    print(os.path.isfile(ocm))
    print(os.path.isfile(outLabel))

    print(1)
    os.system(cmd_call)
    print(2)
        
    
"""
Classifier evaluation methods.
"""

def populationRegionSize(subjectList,labelDirectory,extension):
    
    """
    Method to compute the sizes of each region across the training set.
    
    Parameters:
    - - - - - 
        subjectList : (list,.txt) list of subjects to include
        labelDirectory : directory containing the label files
        extension : label file extension
    """
    
    if isinstance(subjectList,str):
        with open(subjectList,'r') as inFile:
            subjects = inFile.readlines()
        subjects = [x.strip() for x in subjects]
    elif isinstance(subjectList,list):
        subjects = subjectList
    else:
        print 'Incorrect subject list type.'
        
    labelSizes = {k: [] for k in np.arange(1,181)}
    
    for subj in subjects:
        
        inLabel = labelDirectory + str(subj) + extension
        
        if os.path.isfile(inLabel):
            
            pred = ld.loadGii(inLabel)
            
            for k in labelSizes.keys():
                if k in set(pred):
                    labelSizes[k].append(len(np.where(pred == k)[0]))
    
    return labelSizes
        
        
        
        