#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:16:08 2017

@author: kristianeschenburg
"""

import classifierUtilities as cu
import loaded as ld
import matchingLibraries as lb

import inspect
import json

import copy
import h5py
import os
import numpy as np
import sklearn


class Prepare():
    
    """
    Class to prepare data for classifiers.  Originally, I'd thought
    to process the data internally in the classifier, but I found 
    that I was repeating lots of code across classifer types.  Additionally,
    the data preprocessing is not "clean" -- it makes sense to
    encapsulate the the preprocessing outside of the classifiers themselves.
    """
    
    def __init__(self,dataMap,hemisphere,features):
        
        
        # Check to make sure hemisphere is valid
        if hemisphere not in ['R','L']:
            raise ValueError('{} is not a valid hemisphere.'.format(hemisphere))
        else:
            self.hemisphere = hemisphere

        # check feature value
        if not features and not isinstance(features,list):
            raise ValueError('Features cannot be empty.  Must be a list.')
        else:
            self.features = features
            
        if 'object' not in dataMap.keys():
            raise ValueError('No valid object directory specified.')
    
        if 'midline' not in dataMap.keys():
            raise ValueError('No valid midline directory specified.')
            
        if 'matching' not in dataMap.keys():
            raise ValueError('No valid matching directory specified.')
            
        for k in dataMap.keys():
            maps = dataMap[k].items()
            
            if not os.path.isdir(maps[0][0]):
                raise ValueError('{} is not a valid directory.'.format(maps[0][0]))
        
        self.dataMap = dataMap
        
    def setParams(self,**params):
        
        """
        Set parameters of Prepare object, as in the case of reading in
        parameters from a JSON file.
        """
        
        args,_,_,_ = inspect.getargspec(self.__init__)
        
        if params:
            for key in params:
                if key in args:
                    setattr(self,key,params[key])

    def load(self,subjects,training=True):
        
        """
        Parameters:
        - - - - -
            subjects : list of subjects to include in training data
            training : boolean indicating whether this set of subjects is
                        a training set or not.  If training == True, loaded
                        data arrays will include the true label arrays.
        """

        ### CHECK PARAMETERS BEFORE PREPROCESSING ###
        features = self.features
        hemisphere = self.hemisphere
        dataMap = self.dataMap
        
        # Check subject list
        if not subjects and not isinstance(subjects,list):
            raise ValueError('Subjects variable must be a non-empty list.')

        print 'Loading training data with {} features.'.format(features)

        loadingFeatures = copy.deepcopy(features)
        if training:
            loadingFeatures.append('label')

        ### LOAD DATA ###
        
        # dataDictionary will be a dictionary of sub-dictionaries.
        # Super keys are subject names. For each sub-dictionary (super value),
        # keys are feature names, and values are data arrays.
        [dataDictionary,matchDictionary] = loadData(subjects,dataMap,loadingFeatures,hemisphere)

        if not dataDictionary:
            raise ValueError('Data dictionary cannot be empty.')

        if isinstance(dataDictionary,h5py._hl.files.File):
            parseFeatures = copy.deepcopy(features)
            parseFeatures.append('label')

            parsedData = ld.parseH5(dataDictionary,parseFeatures)
            dataDictionary.close()
            dataDictionary = parsedData
        
        return [dataDictionary,matchDictionary]

            
    def training(self,trainData,scale=True,rand=None):
        
        """
        Method to process the training data for each subject.  This includes
        excluding data corresponding to subjects in the test set, randomizing
        the training set to a given size, and applying down-sampling to the
        training data.
        
        Output is the zero-mean, unit variance data array for each training
        subject, as a dictionary.
        
        Parameters:
        - - - - -
            trainData : data to process
            matchData : matches to process
            scale : boolean, whether to fit zero-mean, unit-variance model
                    to data
            exclude_testing : list of subjects to exclude from the training
                                set.  Supply a list here is the same as 
                                removing these subjects from the initial
                                "subjects" list.
            rand : size of training set.  If none, has no effect.  If less
                    than current length of training size, randomly samples 
                    "rand" subjects from training set, after excluding any
                    testing subjects.
        """

        if not isinstance(rand,int) or rand < 1:
            rand = None

        self.rand = rand
        self.scale = scale
        
        features = self.features
        
        nf = []
        for f in features:
            if f != 'label':
                nf.append(f)
        
        ### PROCESS SUBJECTS ###
        
        # get subject IDs in training data
        subjects = trainData.keys()

        # If rand is set, selects min(rand, current_training_size) random 
        # subjects from the current training set.
        if rand and rand < len(subjects):
            sample = np.random.choice(subjects,size=rand,replace=False)
            trainData = {s: trainData[s] for s in sample}
            subjects = trainData.keys()
            
        mergedData = {}.fromkeys(subjects)
        mergedLabels = {}.fromkeys(subjects)

        # For each subject, merge the unique feature arrays into an single
        # array, where ordering of the columns is determined by ordering 
        # of names in the features variable.
        
        supraData = []
        supraLabels = []
        
        for subj in trainData.keys():
            mergedData[subj] = cu.mergeFeatures(trainData[subj],nf)
            mergedLabels[subj] = cu.mergeFeatures(trainData[subj],['label'])
            
            supraData.append(mergedData[subj])
            supraLabels.append(mergedLabels[subj])
        
        supraData = np.squeeze(np.row_stack(supraData))
        supraLabels = np.squeeze(np.concatenate(supraLabels))
        
        labInds = np.where(supraLabels > 0)[0]
        
        # Apply zero-mean, unit-variance scaling AFTER down-sampling
        if self.scale:
            scaler = sklearn.preprocessing.StandardScaler(with_mean=True,
                                                          with_std=True)
            scaler.fit(supraData[labInds,:])
            self.scaler = scaler
        
        for subj in mergedData.keys():
            tempInds = np.where(mergedLabels[subj] != 0)[0]
            
            mergedData[subj][tempInds,:] = scaler.transform(mergedData[subj][tempInds,:])

        return [mergedData,mergedLabels]


    def testing(self,subject):
        
        """
        Method to process testing data.  If the training data was scaled to
        zero-mean and unit variance, we transform the test data using the fit
        training data model.

        Parameters:
        - - - - -
            dataObject : SubjectFeatures object for a test brain      
            matchingMatrix : matrix containg how frequently a given vertex
                                maps to a vertex assigned to each label
        """

        features = self.features
        scaler = self.scaler
        
        print 'Loading testing data with {} features.'.format(features)
        
        objDict = self.dataMap['object'].items()
        objDir = objDict[0][0]
        objExt = objDict[0][1]

        matDict = self.dataMap['matching'].items()
        matDir = matDict[0][0]
        matExt = matDict[0][1]
        
        dataObject = ''.join([objDir,subject,objExt])
        matchingMatrix = ''.join([matDir,subject,matExt])

        # load test subject data, save as attribtues
        rawTestData = ld.loadH5(dataObject,*['full'])
        ID = rawTestData.attrs['ID']
        
        parsedData = ld.parseH5(rawTestData,features)
        rawTestData.close()

        testData = parsedData[ID]
        testData = cu.mergeFeatures(testData,features)

        if self.scale:
            scaler = self.scaler
            mtd = scaler.transform(testData)
            
        threshed = ld.loadMat(matchingMatrix)

        ltvm = cu.vertexMemberships(threshed,180)

        # Threshed : constrained matching matrix
        # mtd : merged test data array
        # ltvm : label-to-vertex maps
        
        return [threshed,mtd,ltvm]


def loadData(subjectList,dataMap,features,hemi):
    
    """
    Generates the training data from a list of subjects.
    
    Parameters:
    - - - - -
        subjectList : list of subjects to include in training set
        dataDir : main directory where data exists -- individual features
                    will exist in sub-directories here
        features : list of features to include
        hemi : hemisphere to process
    """

    objDict = dataMap['object'].items()
    objDir = objDict[0][0]
    objExt = objDict[0][1]
    
    print 'Objects extracted.'
    
    midDict = dataMap['midline'].items()
    midDir = midDict[0][0]
    midExt = midDict[0][1]
    
    print 'Midlines extracted.'
    
    matDict = dataMap['matching'].items()
    matDir = matDict[0][0]
    matExt = matDict[0][1]
    
    print 'Matchings extracted.'

    data = {}
    matches = {}

    for s in subjectList:

        # Training data
        trainObject = '{}{}.{}.{}'.format(objDir,s,hemi,objExt)
        midObject = '{}{}.{}.{}'.format(midDir,s,hemi,midExt)
        matObject = '{}{}.{}.{}'.format(matDir,s,hemi,matExt)

        # Check to make sure all 3 files exist
        if os.path.isfile(trainObject) and os.path.isfile(midObject) and os.path.isfile(matObject):

            # Load midline indices
            # Subtract 1 for differece between Matlab and Python indexing
            mids = ld.loadMat(midObject)-1
            mids = set(mids)
            
            match = ld.loadMat(matObject)

            # Load training data and training labels
            trainH5 = h5py.File(trainObject,mode='r')

            # Get data corresponding to features of interest
            subjData = ld.parseH5(trainH5,features)
            trainH5.close()
            
            nSamples = set(np.arange(subjData[s][features[0]].shape[0]))
            coords = np.asarray(list(nSamples.difference(mids)))
            
            for f in subjData[s].keys():
                tempData = subjData[s][f]
                if tempData.ndim == 1:
                    tempData.shape+=(1,)

                subjData[s][f] = np.squeeze(tempData[coords,:])
                
            match = match[coords,:]
            
            data[s] = subjData[s]
            matches[s] = match

    return [data,matches]

def neighborhoodMaps(neighborhoodMap,neighborhoodType,distance):
    
    """
    Construct the neighborhood mapping information for the training data.
    Specifically, when we are building a classifier to distinguish neighboring 
    regions from one another, we want to determine which regions get confused
    most often.
    
    We consider either the results of the train-train matchings, or we simply
    use the label-label distance matrix.
    
    Parameters:
    - - - - -
        neighborhoodMap : neighborhood map file (Dijkstra distance file, or 
                            MergedMaps file)
        
        neighborhoodType : 'adjacency' (Dijkstra) or 'confusion' (MergedMaps)
        
        distance : threshold (hops for Dijkstra, frequency for MergedMaps)
    """
    
    if not os.path.isfile(neighborhoodMap):
        raise ValueError('Neighborhood map does not exist.')
    
    if neighborhoodType not in ['adjacency','confusion']:
        raise ValueError('Incorrect neighborhood type.')
    
    # load and prepare neighborhoodMap
    neighborhoodMap = ld.loadPick(neighborhoodMap)
    
    if neighborhoodType == 'adjacency':
        boundary = 'inside'
        threshold = 1
    else:
        boundary = 'outside'
        threshold = 0.05
        neighborhoodMap = lb.mappingFrequency(neighborhoodMap)
    
    neighbors = lb.mappingThreshold(neighborhoodMap,threshold,boundary)
    
    return neighbors

def parseJSON(templateFile):
    
    DATA_PARAMETERS = ['features','training','hemisphere',
                       'dataMap','downsample']
    
    CLASSIFIER_PARAMETERS = {'RandomForest': ['max_depth','n_estimators',
                                              'power'],
                            'MALP': ['atlases','atlas_size'],
                            'NeuralNetwork': ['eval_factor','layers',
                                              'node_structure','epochs',
                                              'batch_size','rate']
                            }
    
    assert os.path.isfile(templateFile)
    
    with open(templateFile,'r') as template:
        template = json.load(template)
        
    data = template['data']
    classifier = template['classifier']

    Prep = Prepare(dataMap,hemisphere,features)
    
    return Prep
    
    
    
    
    
    
        
    