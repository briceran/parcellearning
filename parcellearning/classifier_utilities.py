#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:22:13 2017

@author: kristianeschenburg
"""

import loaded as ld
import matchingLibraries as lb

import copy
import h5py
import pickle

import numpy as np
from sklearn import preprocessing

"""
##########

Below, we have methods to help build generic atlases and classifiers.

##########
"""

def buildResponseVector(labels,labelData):
    
    """
    Method to build the response vector.  The response vector is a vector
    of labels, where each index corresponds to the value to
    predict, given a single feature vector.  Generally, the feature vector
    will be "many" vectors, for each data sample.
    
    Parameters:
    - - - - - 
        labels : list of labels in training data
        
        labelData : array of feature data for each label, generally has been
                    concatenated across subjects 
    Returns:
    - - - - 
        response : vector containing training response
    """
    
    response = {}.fromkeys(list(labels))
    
    for r in response.keys():
        
        tempData = labelData[r]
        tempResp = np.repeat(r,tempData.shape[0])
        tempResp.shape += (1,)
        
        response[r] = tempResp
    
    return response
    
def coreData(trainData,core,feats):
    
    """
    Method to collect feature data corresponding to a "core" label.  By core, 
    we mean each unique label across the training data.  For HCP, this will
    generally be 180 unique "core" labels.
    
    Concantenates data for "core" for all features in "feats".
    
    Parameters:
    - - - - - 
        trainData : loaded training data object        
        core : unique label value
        feats : list of features to extract data for
    Returns:
    - - - -
        featsData : array of feature data corresponding to core label value        
    """

    featsData = []

    for subj in trainData:
        
        # get training data for single subject
        # these subjects will not have been included in the data object
        # unless "train" attribute == True, so they will have a "label" 
        # attribute as well
        subjFullData = trainData[subj]
        
        # get subject-specific label data
        label = np.asarray(subjFullData['label'])
        
        # list, containing "core" label data for each feature
        subjFeatData = []
            
        # get indices where core label exists
        #ix = np.in1d(label.ravel(),core).reshape(label.shape)
        ix = np.where(label == core)[0]

        # loop over specified-features
        for f in feats:
            # make sure specified features exist in subject data
            try:
                fData = subjFullData[f]
            except KeyError:
                raise
            else:
                subjFeatData.append(fData[ix,:])

        # stack subject-specific features
        featsData.append(np.column_stack(subjFeatData))
        
    # stack subject features to make matrix 'tall' (Subjects by Features)
    featsData = np.row_stack(featsData)
    
    return featsData

def getLabels(trainData):
        
        """
        Get all unique labels that exist across entire set of training data
        
        Parameters:
        - - - - - 
            trainData : training data object
        Returns:
        - - - - 
            labs : list of all the unique label values in a training object
            
        """
        
        labs = set([])
        
        # loop over all training subjects in training data
        # by construction, will have a feature called 'label'
        for subj in trainData:

            labelData = np.squeeze(trainData[subj]['label'])
            labs.update(set(labelData))
            
        return list(labs)
    
def mergeFeatures(yData,feats,**kwargs):
    
    """
    Method to merge the subject features of interest into a single array.
    
    Parameters:
    - - - - - 
        yData : SubjectFeatures object "data" attribute    
        
        feats : features the be merged together
        
        **kwargs : contains StandardScaler() objects to standardize the 
                    test data using the training data features
    """

    data = []
    
    if kwargs:
        if kwargs['scale']:
            scalers = kwargs['scale']
        else:
            print('Does not have scaler objects -- will return '
                  'non-standardized data.')
    
    for f in feats:
        
        if f in yData.keys():
            featData = yData[f]
            try:
                scalers
            except:
                pass
            else:
                print('scalers exist.')
                featData = scalers[f].transform(featData)
            finally:
                data.append(featData)
        else:
            print('Object does not have feature {}.'.format(f))
        
    data = np.column_stack(data)
    
    return data
    
def mergeLabelData(labelData,responseData,labels):
    
    """
    Method to merge the training data corresponding to a set of labels.
    
    Parameters:
    - - - - - 
        labelData : dictionary where keys correspond to unique
                    labels and values correspond to aggregate data for that 
                    label across all training subjects      
        responseData : dictionary where keys correspond to unique labels and 
                        values correspond to vectos of length N_i samples
                        (for label i)
        labels : labels to aggregate over
        
    Returns:
    - - - -
        learnData : array correspond to "stacked" labelData arrays, where only
                    data corresponding to l in labels are stacked
        y : array corresponding to "stacked" response vectors, where only 
            vectors corresponding to l in labels are stacked
    """
    
    learnData = []    
    y = []
    
    for l in labels:
        if l in labels:
            learnData.append(labelData[l])
            y.append(responseData[l])
        
    learnData = np.row_stack(learnData)
    y = np.row_stack(y)
    
    return(learnData,y)

def parseKwargs(acceptable,kwargs):
    
    """
    Method to check **kwargs input to methods to receive it.  If some of
    the values are set to None, will remove the key that corresponds to it.
    
    Parameters:
    - - - - -
        classInstance : instance of class whose arguments might needed to be 
                        adapted
        kwargs : possible arguments supplied to function
        
    Returns:
    - - - -
        kwargs : supplied key-value arguments that are included in the 
                    classInstance arguments
    """
    
    output = {}

    if kwargs:
        for key in kwargs.keys():
            
            if key in acceptable:
                output[key] = kwargs[key]

    return output

def partitionData(trainData,feats):
    
    """
    Method to parition feature data for all labels, from all training subjects
    in the training object -- partitions the training data features by label.
    
    Parameters:
    - - - - - 
        trainData : loaded training data object
        feats : list of features to whose data to include for each core
        
    Returns:
    - - - - 
        labelData : dictionary where keys are the unique labels in the
                    training data and values are arrays corresponding to the 
                    feature data for the label, aggregated from the entire
                    training data object
    """
    
    labs = set(getLabels(trainData)).difference({0,-1})
    labelData = {}
        
    for l in labs:
        if l > 0:
            
            labelData[l] = coreData(trainData,l,feats)
                
    return labelData

def prepareUnitaryFeatures(trainingObject):
    
    """
    Parameters:
    - - - - -
        trainingObject : single subject training object
    """

    ID = trainingObject.attrs['ID']

    training = {}
    training[ID] = trainingObject.data
    
    return training

def vertexMemberships(labelMaps,labels):
        
    """
    Method to partition vertices based on which labels each
    vertex maps to in the training brains.  As more training brains are
    included in the **MatchingFeaturesTest** object, a vertex might map
    to an increasing variety of labels.
    
    We do this so we can apply a classifier to multiple vertices at the same
    time, rather than looping through each vertex, through all classifiers.
    
    Parameters:
    - - - - - 
        labelMaps : vertex-to-label maps for MatchingLibraryTest object    
        labels : set of labels in training data
        
    Returns:
    - - - - 
        labelVerts : dictionary that maps labels to sets of vertices that 
                        map to it
    """
    
    labelVerts = {}.fromkeys(list(labels))
    
    for L in labels:
        
        # find all vertices that map to label L
        # not exclusive
        inds = [k for k,v in labelMaps.items() if v and L in v]
        # map indices to label-indices pair
        labelVerts[L] = inds
        
    return labelVerts


"""
##########

Below are methods related to the classification or prediction steps.

##########
"""

def combineClassifications(models,ids):
    
    """
    Combines classifier counts from multiple Atlases via list concatenation.
    
    Parameters:
    - - - - -
        models : individual atlases that have "predicted"
        
        ids : names of vertices
        
    Returns:
    - - - -
        counts : update baseline dictioncary with prediction counts from 
                    multiple Atlases
    """
    
    counts = {}.fromkeys(ids)
    
    for m in models:
        for i in ids:
            # if counts[i] doesn't have any values
            if not counts[i]:
                counts[i] = m.baseline[i]
            else:
                counts[i] = counts[i] + m.baseline[i]
                            
    return counts

def combineFilter(mappings,combined,ids):
    
    """
    Filter the classification results to include only those labels that a test
    vertex mapping to using surface registration.
    """
    
    filtered = {}.fromkeys(ids)
    
    for i in ids:
        if combined[i] and mappings[i]:
            
            ic = combined[i].keys()
            im = mappings[i].keys()
            
            filtered[i] = {l: combined[i][l] for l in ic if l in im}
    
    return filtered

def countClassifications(classifications,ids):
    
    """
    Counts instances of each classified element, returns dictionary of counts
    for each test point.
    
    Parameters:
    - - - - - 
        classifications : (dictionary)  Keys are vertices, values are lists
                            of labels
        ids : (list) of vertex IDs
    Returns:
    - - - -
        counts : (dictionary) Keys are vertices, values are sub-dictionaries.
                                Sub-keys are label, sub-values are counts.
    """
    
    counts = {k: {} for k in ids}
    
    for k in ids:
        if classifications[k]:
            for l in set(classifications[k]):
                counts[k].update({l: classifications[k].count(l)})
                
    return counts

def frequencyClassifications(baselineCounts,predicted,ids):
    
    """
    Computes frequency with which test label was classified as final label.
    
    Parameters:
    - - - - -
        baselineCounts : (dictionary) Key are vertices, values are 
                            sub-dictionaries.  Sub-keys are labels, and
                            sub-values are counts.
    Returns:
    - - - -
        maxFreq : (dictionary) Keys are vertices, values are frequencies.
        
    """
    
    pred = copy.copy(predicted)
    
    maxFreq = {}.fromkeys(ids)
    
    freqs = lb.mappingFrequency(baselineCounts)
    
    for n,k in enumerate(ids):
        print(n,k)
        if freqs[k]:
            maxFreq[k] = freqs[k][pred[n]]
    
    return maxFreq

def maximumLiklihood(y,yMatch):
    
    """
    Method to return the label mapped to the most frequently by all test
    vertices.
    
    Parameters:
    - - - - -
        y : loaded SubjectFeatures object for a test brain
        yMatch : loaded MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
    """
    
    verts = np.arange(0,y.obs)
        
    maxLik = {}
    maxLik = maxLik.fromkeys(verts)
        
    for vert in maxLik.keys():
        if vert not in yMatch.mids:
            
            temp = yMatch.vertLib[vert]
            ml = max(temp,key=temp.get)
            maxLik[vert] = ml
    
    return maxLik

def maximumProbabilityClass(mappingMatrix,predMatrix):
    
    """
    Many classifiers output and prediction probability matrix, where each row
    corresponds to a data point, and each column corresponds to a specific
    class.  The value at each index corresponds to the prediction probability
    that a given point belongs to a given class.
    
    This method selects the highest scoring class, given the mappingMatrix.
    
    Parameters:
    - - - - -
        mappingMatrix : binary matrix, where each index is 0 or 1, indicating
                        whether that vertex mapping to that label.
        predMatrix : float matrix, containing probability that a given data
                        point belongs to a given class.
    """
    
    threshed = mappingMatrix*predMatrix;
    
    scores = np.argmax(threshed,axis=1);
    
    return scores;
            
def saveClassifier(classifier,output):
    
    """
    Method to save a classifier object.
    
    Parameters:
    - - - - -
        classifier : for now, just a GMM object of Mahalanobis object.
    """
    if classifier._fitted:
        try:
            with open(output,"wb") as outFile:
                pickle.dump(classifier,outFile,-1);
        except:
            pass
    else:
        print('Classifier has not been trained.  Not saving.')

def standardize(grouped, features):
    """
    Method to demean the data from a GroupFeatures object.  This object is
    just a dictionary of dictionaries -- each main key is a subject ID, with
    sub-keys correpsonding to features i.e. resting-state, cortical metrics.

    Standardization is performed upon run-time -- we might want to save the
    mean and variance of each feature, and will return these, along with a
    demeaned and unit-varianced GroupFeatures object.

    Parameters:
    - - - - -
        grouped : pickle file -- output of GroupFeatures.save(
                    {'train': 'outputFile.p'})
    """

    if isinstance(grouped, str):
        trainData = ld.loadH5(grouped, *['full'])        
    elif isinstance(grouped,h5py._hl.files.File):
        trainData = grouped        
    elif isinstance(grouped,dict):
        trainData = grouped        
    else:
        raise ValueError('Training data cannot be empty.')
        
    subjects = trainData.keys()

    mappings = {}.fromkeys(subjects)
    scalers = {}.fromkeys(features)

    scale = preprocessing.StandardScaler(with_mean=True,with_std=True)

    for f in features:

        c = 0
        tempData = []
        scalers[f] = copy.deepcopy(scale)

        for s in subjects:
            mappings[s] = {}

            subjData = trainData[s][f]
            [x, y] = subjData.shape
            mappings[s]['b'] = c
            mappings[s]['e'] = c + x

            tempData.append(subjData)
            c += x

        tempData = np.row_stack(tempData)
        tempData = scalers[f].fit_transform(tempData)

        for s in subjects:
            coords = mappings[s]
            coordData = tempData[coords['b']:coords['e'],:]
            trainData[s][f] = coordData

    return (trainData, scalers)

def updatePredictions(storage,members,predictions):
    
    """
    Updates dictionary with predicted labels.  Rather than keeping track of 
    prediction probabilities, keeps track of counts with which vertex is 
    classified as beloning to specific label.
    
    Parameters:
    - - - - -
        storage : dictionary in which predictions are stored
        members : vertices mapping to label
        predictions : predicted labels of members, for 'label' core model
    """
    
    stor = copy.copy(storage)
    mem = copy.copy(members)
    pre = copy.copy(predictions)
    
    if isinstance(mem,int):
        vert = mem
        pred = pre[0]
        
        stor[vert].append(pred)
        
    else:
        for vert,pred in zip(mem,pre):
            stor[vert].append(pred)
    
    return stor

def updateScores(storage,label,members,scores):
        
        """
        Updates the dictionary with scores for all vertices that
        map to label.
        
        Parameters:
        - - - - - 
            storage : dictionary in which scores are stored
            label : label of interest
            members : vertices mapping to label
            scores : computed scores of members to feature data
                        of label
        """

        for vert,scr in zip(members,scores):
            # if vertex has not yet been mapped
            # initialize new key-value pair
            if not storage[vert]:
                storage[vert] = {label: scr}
            # otherwise update key
            else:
                storage[vert].update({label: scr})
                
        return storage
    
    
    