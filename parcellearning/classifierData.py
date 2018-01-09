#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:16:08 2017

@author: kristianeschenburg
"""

import classifierUtilities as cu
import dataUtilities as du
import loaded as ld


import copy
import inspect
import os
import numpy as np
from sklearn import preprocessing

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

    def training(self,subjects,training=True,scale=True:
        
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

        loading = copy.deepcopy(nf)
        if training:
            loading.append('label')
        
        # Check subject list
        if not subjects and not isinstance(subjects,list):
            raise ValueError('Subjects variable must be a non-empty list.')
        

        ### LOAD DATA ###
        
        # dataDictionary will be a dictionary of sub-dictionaries.
        # Super keys are subject names. For each sub-dictionary (super value),
        # keys are feature names, and values are data arrays.
        print 'Training data columns (in order): {}'.format(loading)
        [dataDict,matchDict] = cu.loadData(subjects,dataMap,loading,hemisphere)

        if not dataDict:
            raise ValueError('Data dictionary cannot be empty.')
        else:
            parsedData = ld.parseH5(dataDict,loading)
            dataDict = parsedData
            
        ### At this point, data has been loaded.
        ### It exists as a dictionary of dictionaries
        
        # get subject IDs in training data
        subjects = dataDict.keys()

        mergedData = {}.fromkeys(subjects)
        mergedLabels = {}.fromkeys(subjects)
        
        # For each subject, merge the unique feature arrays into an single
        # array, where ordering of the columns is determined by ordering 
        # of names in the features variable.
        print 'Keys to merge: {}'.format(nf)
    	print 'Keys possible: {}\n'.format(loading)

    	count = 0
    	test = dataDict[dataDict.keys()[0]]
    	for f in nf:
    		try:
    			fShape = test[f].shape[1]
    		except:
    			fShape = np.ndim(test[f])
    		count += fShape
    		print '{} shape: {}'.format(f,fShape)

        for subj in dataDict.keys():
            mergedData[subj] = du.mergeFeatures(dataDict[subj],nf)
            mergedLabels[subj] = du.mergeFeatures(dataDict[subj],['label'])
            
        supraData = du.mergeValueArrays(mergedData)
        supraLabels = du.mergeValueLists(mergedLabels)
    	
    	print '\nExpected feature matrix dimensionality: {}'.format(count)
    	print 'Computed feature matrix dimensionality: {}\n'.format(supraData.shape[1])
    	assert count == supraData.shape[1]
    	
    	print 'Computed feature matrix samples: {}'.format(supraData.shape[0])
    	print 'Computed response matrix samples: {}\n'.format(supraLabels.shape[0])
    	assert supraData.shape[0] == supraLabels.shape[0]
     
    	labInds = np.where(supraLabels != 0)[0]

        # Apply zero-mean, unit-variance scaling
        if self.scale:
            
            print 'Standardizing samples.'
            scaler = preprocessing.StandardScaler(with_mean=True,
                                                          with_std=True)
            scaler.fit(supraData[labInds,:])
            self.scaler = scaler
        
        for subj in mergedData.keys():
            tempInds = np.where(mergedLabels[subj] > 0)[0]
            
            mergedData[subj][tempInds,:] = scaler.transform(mergedData[subj][tempInds,:])

        return [mergedData,mergedLabels,matchDict]
    
    
def testing(prepared,subject,trDir=None,trExt=None,
            mtDir=None,mtExt=None):
    
    """
    Method to process testing data.  If training data was standardized,
    we transform the test data using the fitted training model.

    Parameters:
    - - - - -
        subject : name of subject to load.  Uses same data map as 
                    training data, so file format must be the same.
        trDir : directory where testing objects exist (defaults to dataMap)
        teExt : extension of testing objects (defaults to dataMap)
    Returns:
    - - - -
        x_test : test data for subject
        matchingMatrix : thresholded matching matrix for subject.  
                            The threshold is set to a minimum of 0.05.
        ltvm : label-to-vertex mapping dictionary.  Keys are label IDs,
                and values are lists of vertices that map to this label.
    """

    if not trDir:
        objDir = prepared.dataMap['object'].keys()[0]
    else:
        objDir = trDir

    if not trExt:
        objExt = prepared.dataMap['object'].values()[0]
    else:
        objExt = trExt

    if not mtDir:
        matDir = prepared.dataMap['matching'].keys()[0]
    else:
        matDir = mtDir

    if not mtExt:
        matExt = prepared.dataMap['matching'].values()[0]
    else:
        matExt = mtExt

    features = prepared.features
    hemisphere = prepared.hemisphere
    scaler = prepared.scaler

    print 'Test data columns (in order): {}'.format(features)

    dataObject = '{}{}.{}.{}'.format(objDir,subject,hemisphere,objExt)
    matchingMatrix = '{}{}.{}.{}'.format(matDir,subject,hemisphere,matExt)

    # load test subject data, save as attribtues
    rawTestData = ld.loadH5(dataObject,*['full'])
    ID = rawTestData.attrs['ID']
    
    parsedData = ld.parseH5(rawTestData,features)
    rawTestData.close()

    testData = parsedData[ID]
    testData = du.mergeFeatures(testData,features)

    if prepared.scale:
        scaler = prepared.scaler
        x_test = scaler.transform(testData)
        
    matchingMatrix = ld.loadMat(matchingMatrix)
    ltvm = cu.vertexMemberships(matchingMatrix,180)

    return [x_test,matchingMatrix,ltvm]
