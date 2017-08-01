#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:54:26 2017

@author: kristianeschenburg
"""

from copy import deepcopy

import classifier_utilities as cu
import featureData as fd
import matchingLibraries as lb
import loaded as ld

import copy
import inspect
import numpy as np
import os
import pickle

from sklearn import covariance, neighbors, mixture

##########
##########
##########
##########

class Mahalanobis(object):
    
    """
    Mahalanobis Class to compute the Mahalanobi distance between test data 
    and training data in order to classify the new data.
    
    Parameters:
    - - - - - -
    
        trainObj : GroupFeatures object containing training data and features 
                    of interest
    """
    
    def __init__(self,trainObj,feats):
        
        if isinstance(trainObj,str):
            self._trainData = ld.loadPick(trainObj)
        elif isinstance(trainObj,dict):
            self._trainData = trainObj
            
            
        if not self._trainData:
            raise ValueError('Training data cannot be empty.')
        
        if not feats:
            raise ValueError('Feature list cannot be empty.')
        else:
            self._features = feats

        self._labels = set(cu.getLabels(self._trainData)) - set([0,-1])
        
        self._labelData = cu.partitionData(self._trainData,feats = feats)
        
        self._mu = self._computeMeans()
        
    def fit(self, covObj = covariance.ShrunkCovariance(shrinkage=0.2)):
        
        """
        
        Generates covariance matrices for each label.  We can use a variety of
        covariance estimates here:
            
            empirical (unadvised)
            shrunken
                Basic
                Ledoit-Wolf
                Oracle
            sparse (via graphical lasso)
            robust (via minimum covariance determinant)
            
        Parameters:
        - - - - - 
            
            covObj : covariance estimation object
                    User specifies what type of covariance estimation they
                    want to incorprate by providing a object

        """
        
        cov = {}
        cov = cov.fromkeys(self._labels)

        precision = {}
        precision = precision.fromkeys(self._labels)
        
        for l in self._labels:

            cov[l] = deepcopy(covObj)
            cov[l].fit(self._labelData[l])
            precision[l] = cov[l].get_precision()
            
        self._cov = cov
        self._precision = precision
        self._fitted = True
  
    def predict(self,y,yMatch,features):
        
        """
        Method to compute Mahalanobis distance of test data from the
        distribution of all training data for each label.
        
        Parameters:
        - - - - - 
        
            y : SubjectFeatures object for a test brain
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
                    
            feats : names of features to include in the distance calculations

        """

        # load SubjectFeatures object
        testObject = ld.loadPick(y)
        
        # load MatchingLibraryTest
        testMatch = ld.loadPick(yMatch) 
        lMaps = testMatch.vertLib
        simplified = {n: lMaps[n].keys() for n in lMaps.keys()}
        
        # Merge the feature data for the test data into single array and
        # compute which vertices map to which labels
        mergedData = cu.mergeFeatures(testObject.data,features)
        
        # this is time consuming -- don't recompute every time if wanting
        # to run prediction with multiple feature sets
        if not hasattr(self,'_labelVerts'):
            labelVerts = cu.vertexMemberships(simplified,self._labels)
            self._labelVerts = labelVerts
        else:
            labelVerts = self._labelVerts
            
        # initialize Mahalanobis prediction vector
        predict = {}
        predict = predict.fromkeys(testMatch.vertLib.keys())

        # for all labels in in training set
        for lab in self._labels:

            # compute vertices that map to that label
            members = labelVerts[lab]
                        
            if len(members) > 0:
                # compute Mahalanobis distane of vertex feature to label feaatures
                scores = self._predictPoint(mergedData,lab,members)
                
                # save results in self.predict
                predict = cu.updateScores(predict,lab,members,scores)
        
        self._predict = predict

    def _predictPoint(self,data,label,members):
        
        """
        Compute the Mahalanobis distances for all vertices mapped to a label.
        
        Parameters:
        - - - - - 
        
            data : merged data array
            
            label : label of interest
        
            members : vertices mapping to label
   
        """

        # get feature data of vertices
        ixData = data[members,:]
        
        # get covariance matrix for label
        #c = self.cov[label].covariance_
        p = self._precision[label]
        # get feature data for label
        #labelData = self.labelData[label]
        muData = self._mu[label]

        # initialize  distance object
        dist = neighbors.DistanceMetric.get_metric('mahalanobis', VI=p)
        
        # compute Mahalanobis distance of vertex features to all label
        # features, take mean
        #mh = np.mean(dist.pairwise(ixData,labelData),1)
        mh = dist.pairwise(ixData,muData)
        
        # return mean Mahalanobis distance for each vertex
        return mh
                
    def weight(self,power):
        
        """
        Method to weight the score by the frequency with which
        the test vertex mapped to the training labels.
        """
        
        vertLib = self._testMatch.vertLib
    
        # for each vertex in test brain, and its mapped labels
        for vert,mapped in vertLib.items():
            vert = np.int(vert)
            
            # if vertex is mapped to labels
            if mapped:

                # get Mahalanobis distances for each label
                # vertex is mapped to
                mahals = self._predict[vert]

                # return weighted distances
                self._weighted[vert] = self._weightPoint(mapped,mahals,power)
            
    def _weightPoint(self,mapLabels,labelDist,power):
        
        """
        Method to weight the scores of a single vertex.
        
        If weight != 0, we weight the distances by 
            
            1 / ( (map count / sum counts)**power)
            
        Since we want the minimum Mahalanobis distance, weighting by the
        frequency will downweight labels that are mapped to most frequently
        (which we don't want) -- we believe that labels mapped to most
        frequently will be correct the correct label, but we want to allow for 
        other possibilities.
        
        Parameters:
        - - - - -
        
            mapLabels : dictionary of labels to which the test vertex is mapped
                        and frequency with which those labels are mapped to
                        
            labelDist : dictionary of labels to which a test vertex is mapped
                        and Mahalanobis distance between vertex test point and 
                        distribution of the feature data for that label
                        
            power : power to which the raise the mapping frequency to
                        
        Returns:
        - - - -
            
            weight : dictionary of weighted Mahalanobis distance for each of 
                        the mapped-to labels
        """
        
        mappedSum = np.float(np.sum(mapLabels.values()))
        labels = mapLabels.keys()
        
        weight = {}

        for l in labels:
            
            if l > 0:
                
                # get mapping frequency
                inv = (1/mappedSum)*mapLabels[l]
                # raise frequency to power
                lWeight = np.power(inv,power)
                # apply weight to Mahalanobis distance
                weight[l] = 1/(lWeight*labelDist[l])
                
        return weight

    def classify(self,weighted=False):
        
        """
        Returns the label for which the mean Mahalanobis distance was 
        the smallest.
        """
        
        if weighted:
            predict = self._weighted
        else:
            predict = self._predict
        
        classes = {}
        classes = classes.fromkeys(predict.keys())
        
        for v,m in predict.items():
            if m:
                classes[v] = min(m,key=m.get)
            else:
                classes[v] = 0
        
        self._classes = classes.values()
        
    def _computeMeans(self):
        
        """
        Method to compute means of each label data set
        """
        
        mu = {}
        mu = mu.fromkeys(self._labels)
        
        for l in mu.keys():
            
            labelData = self._labelData[l]
            
            mu[l] = self._singleMean(labelData)
        
        return mu
        
    
    def _singleMean(labelData):
        
        """
        Compute mean of single label.
        """
        
        m = np.mean(labelData,0)
        
        if len(m.shape) == 1:
            m.shape += (1,)
        
        [x,y] = m.shape
        
        if x > y:
            m = m.T
        
        return m