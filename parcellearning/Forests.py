#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:17:38 2017

@author: kristianeschenburg

"""

import AtlasUtilities as au
import classifier_utilities as cu
import dataUtilities as du

import copy
import inspect

import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# number of cores to parallelize over

class Forest(object):
    
    """
    Class to instantiate a classifier for a single training set.
    
    Parameters:
    - - - - -
        n_estimators : number of estimating trees per forest
        
        max_depth : maximum estimating tree depth
    
    """
    
    def __init__(self,n_estimators = 60, max_depth = 5):
        
        if not n_estimators or n_estimators < 1:
            raise ValueError('The number of estimators must be a positive integer.')
            
        if not max_depth or max_depth < 1:
            raise ValueError('Tree must be at least a depth of 1.')

        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def set_params(self,**kwargs):
        
        """
        Update parameters with user-specified dictionary.
        """
        
        args, varargs, varkw, defaults = inspect.getargspec(self.__init__)
        
        if kwargs:
            for key in kwargs:
                if key in args:
                    setattr(self,key,kwargs[key])
            
    def fit(self, x_train, y_train, neighbors, L, classifier=None,
            model_type='ori',**kwargs):
        
        """
        Method to initialize training data and fit the classifiers.
        
        Parameters:
        - - - - -

            x_train : training feature data, partitioned by response
            
            y_train : training response vectors, partitioned by response
            
            model_type : type of classification scheme for multi-class 
                         A   prediction models

            kwargs : optional arguments for classifier
        """
        
        if not classifier:
            classifier = rfc(n_estimators=self.n_estimators,
                             max_depth=self.max_depth,n_jobs=-1)
            
        
        labels = np.arange(1,L+1)
        self.labels = labels
        self.neighbors = neighbors
        
        x_train = du.mergeValueArrays(x_train)
        y_train = du.mergeValueLists(y_train)
        
        self.input_dim = x_train.shape[1]

        labelKeys = x_train.keys()

        # get valid arguments for supplied classifier
        # get valid parameters passed by user
        # update classifier parameters
        # save base models
        classifier_params = inspect.getargspec(classifier.__init__)
        classArgs = cu.parseKwargs(classifier_params,kwargs)
        classifier.set_params(**classArgs)
        
        print 'depth: {}'.format(classifier.max_depth)
        print 'nEst: {}'.format(classifier.n_estimators)
            
        model_selector = {'oVo': OneVsOneClassifier(classifier),
                          'oVr': OneVsRestClassifier(classifier),
                          'ori': classifier}
        
        models = {}.fromkeys(labels)

        for i,lab in enumerate(labels):
            if lab in labelKeys and lab in neighbors.keys():
                
                # compute confusion set of labels
                labelNeighbors = set([lab]).union(neighbors[lab]).intersection(labels)

                # copy the model (due to passing by object-reference)
                models[lab] = copy.deepcopy(model_selector[model_type])

                # extract data for confusion set, train model
                training = du.mergeValueArrays(x_train,keys = labelNeighbors)
                response = du.mergeValueLists(y_train,keys = labelNeighbors)
                
                models[lab].fit(training,np.squeeze(response))

        self.models = models

    def predict(self,x_test,match,ltvm,softmax_type = 'BASE',power=1):
        
        """
        Method to predict labels of test data.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain      
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
                    
        """

        neighbors = self.neighbors
        input_dim = self.input_dim

        funcs = {'BASE': au.baseSoftMax,
                 'TREES': au.treeSoftMax,
                 'FORESTS': au.forestSoftMax}
        
        labels = self.labels
        R = len(labels)

        [test_samples,test_dim] = x_test.shape
        if test_dim != input_dim:
            raise Warning('Test data does not have the same number \
                          features as the training data.')

        # initialize prediction dictionary
        baseline = np.zeros((test_samples,R+1))


        if power == None:
            match = np.power(match,0)
        elif power == 0:
            nz = np.nonzero(match)
            match[nz] = 1
        else:
            match = np.power(match,power)

        for lab in labels:
            if lab in neighbors.keys():
                
                mapped = ltvm[lab]
                mappedData = x_test[mapped,:]
                estimator = self.models[lab]
                
                if len(mapped) > 0:
                    preds = funcs[softmax_type](estimator,mapped,mappedData,match,R)
                    baseline = cu.updatePredictions(baseline,mapped,preds)
                
        predicted = np.argmax(baseline,axis=1)
        
        self.baseline = baseline
        self.predicted = predicted
        
        return [baseline,predicted]
