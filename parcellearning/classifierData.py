#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:16:08 2017

@author: kristianeschenburg
"""

import classifierUtilities as cu
import dataUtilities as du

import loaded as ld
import matchingLibraries as lb

import inspect

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
    encapsulate the preprocessing outside of the classifiers themselves.
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
            
            noLabel = []
            for f in features:
                if f != 'label':
                    noLabel.append(f)

            self.features = noLabel
            
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

    def training(self,subjects,training=True,scale=True):
        
        """
        Parameters:
        - - - - -
            subjects : list of subjects to include in training data
            training : boolean indicating whether this set of subjects is
                        a training set or not.  If training == True, loaded
                        data arrays will include the true label arrays.
        Returns:
        - - - -
            mergedData : standardized, merged feature data arrays for each 
                            subj in subjects.  
            mergedLabels : dependent-variable vector of vertex labels for 
                            each subj in subjects.  Number of samples is the
                            same as the number of samples in mergedData.
            matchDictionary : matching matrix for each subj in subjects,
                                where each row is the mapping vertex to label
                                mapping vector.
        """
        
        ### Check Parameters before loading any data
        self.scale = scale

        ### CHECK PARAMETERS BEFORE PREPROCESSING ###
        features = self.features
        hemisphere = self.hemisphere
        dataMap = self.dataMap
        
        nf = []
        for f in features:
            if f != 'label':
                nf.append(f)
                
        loadingFeatures = copy.deepcopy(nf)
        if training:
            loadingFeatures.append('label')
        
        # Check subject list
        if not subjects and not isinstance(subjects,list):
            raise ValueError('Subjects variable must be a non-empty list.')
        

        ### LOAD DATA ###
        
        # dataDictionary will be a dictionary of sub-dictionaries.
        # Super keys are subject names. For each sub-dictionary (super value),
        # keys are feature names, and values are data arrays.
        print 'Training data columns (in order): {}'.format(nf)
        [dataDictionary,matchDictionary] = loadData(subjects,dataMap,loadingFeatures,hemisphere)

        if not dataDictionary:
            raise ValueError('Data dictionary cannot be empty.')
        else:
            parseFeatures = copy.deepcopy(nf)
            parseFeatures.append('label')
    
            parsedData = ld.parseH5(dataDictionary,parseFeatures)
            dataDictionary = parsedData
            
        ### At this point, data has been loaded.
        ### It exists as a dictionary of dictionaries
        
        # get subject IDs in training data
        subjects = dataDictionary.keys()

        mergedData = {}.fromkeys(subjects)
        mergedLabels = {}.fromkeys(subjects)
        
        # For each subject, merge the unique feature arrays into an single
        # array, where ordering of the columns is determined by ordering 
        # of names in the features variable.

        for subj in dataDictionary.keys():
            mergedData[subj] = du.mergeFeatures(dataDictionary[subj],nf)
            mergedLabels[subj] = du.mergeFeatures(dataDictionary[subj],['label'])
            
        supraData = du.mergeValueArrays(mergedData)
        supraLabels = du.mergeValueLists(mergedLabels)

        labInds = np.where(supraLabels != 0)[0]

        # Apply zero-mean, unit-variance scaling
        if self.scale:
            
            print 'Standardizing samples.'
            scaler = sklearn.preprocessing.StandardScaler(with_mean=True,
                                                          with_std=True)
            scaler.fit(supraData[labInds,:])
            self.scaler = scaler
        
        for subj in mergedData.keys():
            tempInds = np.where(mergedLabels[subj] > 0)[0]
            
            mergedData[subj][tempInds,:] = scaler.transform(mergedData[subj][tempInds,:])

        return [mergedData,mergedLabels,matchDictionary]
    
    
    def testing(self,subject):
        
        """
        Method to process testing data.  If training data was standardized,
        we transform the test data using the fitted training model.

        Parameters:
        - - - - -
            subject : name of subject to load.  Uses same data map as 
                        training data, so file format must be the same.
        Returns:
        - - - -
            matchingMatrix : thresholded matching matrix for subject.  
                                The threshold is set to a minimum of 0.05.
            x_test : test data for subject
            ltvm : label-to-vertex mapping dictionary.  Keys are label IDs,
                    and values are lists of vertices that map to this label.
        """

        features = self.features
        hemisphere = self.hemisphere
        scaler = self.scaler
        
        nf = []
        for f in features:
            if f != 'label':
                nf.append(f)
        
        print 'Test data columns (in order): {}'.format(nf)
        
        objDict = self.dataMap['object'].items()
        objDir = objDict[0][0]
        objExt = objDict[0][1]

        matDict = self.dataMap['matching'].items()
        matDir = matDict[0][0]
        matExt = matDict[0][1]
        
        dataObject = '{}{}.{}.{}'.format(objDir,subject,hemisphere,objExt)
        matchingMatrix = '{}{}.{}.{}'.format(matDir,subject,hemisphere,matExt)

        # load test subject data, save as attribtues
        rawTestData = ld.loadH5(dataObject,*['full'])
        ID = rawTestData.attrs['ID']
        
        parsedData = ld.parseH5(rawTestData,nf)
        rawTestData.close()

        testData = parsedData[ID]
        testData = cu.mergeFeatures(testData,nf)

        if self.scale:
            scaler = self.scaler
            x_test = scaler.transform(testData)
            
        matchingMatrix = ld.loadMat(matchingMatrix)

        ltvm = cu.vertexMemberships(matchingMatrix,180)

        # matchingMatrix : constrained matching matrix
        # x_test : merged test data array
        # ltvm : label-to-vertex maps
        
        return [x_test,matchingMatrix,ltvm]


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

    midDict = dataMap['midline'].items()
    midDir = midDict[0][0]
    midExt = midDict[0][1]

    matDict = dataMap['matching'].items()
    matDir = matDict[0][0]
    matExt = matDict[0][1]

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