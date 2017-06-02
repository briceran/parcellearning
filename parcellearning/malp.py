#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:17:38 2017

@author: kristianeschenburg
"""
import classifier_utilities as cu
import featureData as fd
import matchingLibraries as lb
import loaded as ld

import copy
import inspect
import numpy as np
import os
import pickle

from joblib import Parallel, delayed
import multiprocessing

from sklearn import ensemble
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# number of cores to parallelize over
NUM_CORES = multiprocessing.cpu_count()
# Valid prediction key-value parameters
PREDICTION_KEYS = ['BASE','TREES','FORESTS']


class Atlas(object):
    
    """
    Class to instantiate a classifier for a single training set.
    
    Parameters:
    - - - - -
        feats : names of features to included in the models
        
        scale : standardized the training data (default == True)
        
        thresh_train : threshold to apply to training subject matches
        
        thresh_test : threshold to apply to test subject matches
        
        softmax_type : type of classification constraining based on surface
                        registration mappings.  BASE for none, TREES at the
                        tree level, and FORESTS at the forest level


        
        random : number of training subjects to include in model
        
        load : if test subject label-verte memberships has been generated
                previously, can provide file
                
        save : path to save label-vertex memberships

    """
    
    def __init__(self,feats,scale=True,thresh_train = 0.05,thresh_test = 0.05,
                 softmax_type='BASE',exclude_testing=None,
                 random=None,load=None,save=None):

        if not feats:
            raise ValueError('Feature list cannot be empty.')
            
        if scale not in [True, False]:
            raise ValueError('Scale must be boolean.')
            
        if thresh_train < 0 or thresh_train > 1:
            raise ValueError('Threshold value must be within [0,1].')
            
        if thresh_test < 0 or thresh_test > 1:
            raise ValueError('Threshold value must be within [0,1].')
            
        if softmax_type not in PREDICTION_KEYS:
            raise ValueError('softmax_type must be either BASE,TREES, or FORESTS.')
            
        if exclude_testing is not None and not isinstance(exclude_testing,str):
            raise ValueError('exclude_testing must by a string or None.')
            
        if random is not None and random < 0:
            raise ValueError('Random must be a positive integer or None.')
            
        if not load is None and not isinstance(load,str):
            raise ValueError('load must be a string or None.')
            
        if save is not None and not isinstance(save,str):
            raise ValueError('save must be a string or None.')

        self.features = feats
        self.scale = scale
        self.thresh_train = thresh_train
        self.thresh_test = thresh_test
        self.softmax_type = softmax_type
        self.exclude_testing = exclude_testing
        self.random = random
        self.load = load
        self.save = save
        
    def set_params(self,**kwargs):
        
        """
        Update parameters with user-specified dictionary.
        """
        
        args, varargs, varkw, defaults = inspect.getargspec(self.__init__)
        
        if kwargs:
            for key in kwargs:
                if key in args:
                    setattr(self,key,kwargs[key])
            
    def fit(self, trainObject, mergedMaps, model_type = 'ori',
            classifier = ensemble.RandomForestClassifier(n_jobs=-1),**kwargs):
        
        """
        Method to initialize training data and fit the classifiers.
        
        Parameters:
        - - - - -
            trainObject : training data (either '.p' file, or dictionary)
            
            mergedMaps : merged MatchingLibraries corresponding to training data
            
            model_type : type of classification scheme for multi-class 
                         A   prediction models
                            
            classifier : model to use

            kwargs : optional arguments for classifier

        """

        threshold = self.thresh_train
        
        self.model_type = model_type
        self._initializeTraining(trainObject,mergedMaps)

        # get valid arguments for supplied classifier
        # get valid parameters passed by user
        # update classifier parameters
        # save base models
        classifier_params = inspect.getargspec(classifier.__init__)
        classArgs = cu.parseKwargs(classifier_params,kwargs)
        classifier.set_params(**classArgs)
        self.classifier = classifier
            
        model_selector = {'oVo': OneVsOneClassifier(classifier),
                          'oVr': OneVsRestClassifier(classifier),
                          'ori': classifier}
        models = {}

        mgm = self.mergedMappings
        freqs = lb.mappingFrequency(mgm)

        c = 1

        for i,l in enumerate(self.labels):
            if l in self.labelData.keys():
                
                # copy the model (due to passing by reference)
                models[l] = copy.deepcopy(model_selector[model_type])

                # threshold the mapping frequencies
                mapped = lb.mappingThreshold(freqs[l],threshold)
                mapped.append(l)
                mapped = list(set(mapped).intersection(self.labels))
                
                # build classifier training data upon request
                [learned,y] = cu.mergeLabelData(self.labelData,self.response,mapped)
    
                models[l].fit(learned,np.squeeze(y))

        self.models = models
        self._fit = True
        
    def _initializeTraining(self,trainObject,mergedMaps):
        
        """
        Initialize the object with the training data.
        
        Parameters:
        - - - - -
            trainObject : input training data (either '.p' file, or dictionary)
            
            mergedMaps : merged MatchingLibraries corresponding to training data
        """

        ## Load / process the training data ##
        
        # if trainObject is filename
        if isinstance(trainObject,str):
            trainData = ld.loadPick(trainObject)
            
            # if trainData is SubjectFeatures object (single subject)
            if isinstance(trainData,fd.SubjectFeatures):
                self.trainingID = trainData.ID
                trainData = cu.prepareUnitaryFeatures(trainData)
                
        # otherwise, if provided with dictionary
        elif isinstance(trainObject,dict):
            trainData = trainObject
        else:
            raise ValueError('Training object is of incorrect type.')
            
        if not trainData:
            raise ValueError('Training data cannot be empty.')
            
        subjects = trainData.keys()
        
        if self.exclude_testing:
            subjects = set(subjects)-set(self.exclude_testing)
        
        # if random is set, sample subsect of training data (no replacement)
        # otherwise, set sample to number of training subjects
        if not self.random:
            randomSample = len(subjects)
        else:
            randomSample = min(self.random,len(subjects))

        sample = np.random.choice(subjects,size=randomSample,replace=False)
        sampleData = {s: trainData[s] for s in sample}
        trainData = sampleData

        # Scale the training data.
        if self.scale:
            [trainData,scalers] = fd.standardize(trainData,self.features)
            
            # scaler objects to transform test data
            self.scalers = scalers
            self._scaled = True

        # get unique labels in training set
        self.labels = set(cu.getLabels(trainData)) - set([0,-1])
        
        # aggregate data corresponding to each label
        self.labelData = cu.partitionData(trainData,feats = self.features)
        
        # build response vector for each set of label data
        self.response = cu.buildResponseVector(self.labels,self.labelData)
        self.mergedMappings = ld.loadPick(mergedMaps)
        
        cond = True
        if not self._compareTrainingDataKeys():
            print('WARNING: Training data and response vectors do not have the same keys.')
            cond = False

        if not self._compareTrainingDataSize():
            print('WARNING: Training data and response vectors are not the same length.')
            cond = False
            
        if not cond:
            raise ValueError('Training data is flawed.')
                    
    def predict(self,y,yMatch):
        
        """
        Method to predict labels of test data.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain      
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
                    
        """
        
        softmax_type = self.softmax_type

        # load the testing data
        self._loadTest(y,yMatch)

        # get names of test data vertices
        verts = self.testMatch.vertLib.keys()
        # get test data
        mtd = self.mergedTestData

        # initialize prediction dictionary
        baseline = {k: [] for k in verts}
        
        # check to see what type of processing option was provided
        if softmax_type == 'BASE':
            for lab in self.labels:

                # compute vertices that map to that label
                members = self.labelToVertexMaps[lab]
                            
                if len(members) > 0:
    
                    # compute member labels from lab core classifier
                    scores = self._predictPoint(mtd,lab,members) 
    
                    # save results in self.predict
                    baseline = cu.updatePredictions(baseline,members,scores)
                    
        else:
            for lab in self.labels:
                members = self.labelToVertexMaps[lab]
                
                if len(members) > 0:
                    
                    if softmax_type == 'TREES':
                        
                        predLabs = np.zeros((len(members),1))
                        
                        predLabs = treeSoftMax(self.models[lab],
                                               self.mappingsCutoff,
                                               members,
                                               mtd[members,:])
                                
                    elif softmax_type == 'FORESTS':
                        
                        predLabs = forestSoftMax(self.models[lab],
                                                        self.mappingsCutoff,
                                                        members,
                                                        mtd[members,:])

                    predictedLabels = np.squeeze(predLabs)
                    baseline = cu.updatePredictions(baseline,
                                                    members,predictedLabels)

        self.baseline = baseline
        
        self.predicted = self._classify(baseline)
        self._classified = True

    def _predictPoint(self,data,label,members):
        
        """
        Compute predicted label of test vertices with specific mapping style.
        
        Parameters:
        - - - - - 
            data : merged data array
            
            label : label of interest
        
            members : vertices mapping to label
        """
        
        # get feature data of vertices
        ixData = data[members,:]

        predictedLabels = self.models[label].predict(ixData)

        predictedLabels = np.squeeze(predictedLabels)

        return predictedLabels
    
    def _classify(self,storage):
        
        """
        For each vertex, returns label that it is classified as most frequently.
        
        Parameters:
        - - - - -
            storage : dictionary of classification label counts
        """

        predicted = {}.fromkeys(storage.keys())

        for v,m in storage.items():
            if m:
                predicted[v] = max(set(m),key=m.count)
            else:
                predicted[v] = 0
                
        return predicted.values()
    
    def _loadTest(self,y,yMatch):
        
        """
        Method to load the test data into the object.  We might be interested
        in loading new test data into, so we have explicitly defined this is
        as a method.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain      
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data

        """
        
        # get Atlas attributes
        threshold = self.thresh_test
        load = self.load
        save = self.save
        
        labels = self.labels
        features = self.features
 
        # load test subject data, save as attribtues
        testObject = ld.loadPick(y)        
        testMatch = ld.loadPick(yMatch)
        
        self.testMatch = testMatch
        self.testObject = testObject
        
        # if the training data has been scaled, apply scaling 
        # transformation to test data and merge features
        data = testObject.data
        
        if self._scaled:
            scalers = self.scalers
            
            for feat in features:
                if feat in data.keys():
                    data[feat] = scalers[feat].transform(data[feat])
                    
        self.mergedTestData = cu.mergeFeatures(data,features)

        # generate a label : vertex dictionary, the vertex contains all 
        # vertices that mapped to label via surface registration
        
        # get vertex to label mapping counts
        vTLM = copy.deepcopy(testMatch.vertLib)
        
        # convert vertex to label mapping counts to frequencies
        freqMaps = lb.mappingFrequency(vTLM)
        
        # if threshold value is greater than 0, there might be labels 
        # that will be cutoff -- otherwise, none will be
        if threshold > 0:
            threshed = {}.fromkeys(freqMaps.keys())
            
            for k in threshed.keys():
                if freqMaps[k]:
                    threshed[k] = lb.mappingThreshold(freqMaps[k],threshold)
                    
            freqMaps = threshed
            
        self.mappingsCutoff = freqMaps
        
        # Computing label-vertex memberships is time consuming
        # If already precomputed for given test data at specified threshold,
        # can supply path to load file.
        if load:
            if os.path.isfile(load):
                self.labelToVertexMaps = ld.loadPick(load)
        
        # Otherwise, compute label-vertex memberships.
        else:
            self.labelToVertexMaps = cu.vertexMemberships(freqMaps,labels)

        # if save is provided, save label-vertex memberships to file
        if save:
            try:
                with open(save,"wb") as outFile:
                    pickle.dump(self.labelToVertexMaps,outFile,-1)
            except IOError:
                print('Cannot save label-vertex memberships to file.')

        self._loadedTest = True
    
    def _compareTrainingDataSize(self):
        
        """
        Method to ensure that the length of the response vector is the same 
        length as the number of observations in the training feature data.
        
        This must be true in order to actually train the classifiers for each
        label.
        """
        cond = True
        
        labelData = self.labelData
        response = self.response
        
        for f,r in zip(set(labelData.keys()),set(response.keys())):
            
            sf = labelData[f].shape[0]
            sr = response[r].shape[0]
            
            if sf != sr:
                cond = False
        
        return cond
            
    def _compareTrainingDataKeys(self):
        
        """
        Method to ensure that the keys for the training data for the response
        vectors are the same.  These must be the same in order to properly
        access the training data for training the classifiers.
        """
        
        labelData = self.labelData
        response = self.response
        
        sf = set(labelData.keys())
        sr = set(response.keys())
        
        return sf == sr
    
    def save(self,outFile):
        
        """
        Method to save the classification object.  This can be saved at any
        time.  However, we can only "test" the classification object if it
        has been trained.
        """
        
        if not self._fit:
            print("Warning: Atlas has not been trained yet.")
        try:
            with open(outFile,"wb") as output:
                pickle.dump(self.models,output,-1)
        except:
            pass
        
def forestSoftMax(metaEstimator,mappings,members,memberData):
    
    """
    Method to restrict whole random decision forest soft-max prediction to
    labels generated in the surface registration step.
    
    Parameters:
    - - - - -
        
        metaEstimator : single decision forest
        
        mappings : vertex to label mappings from surface registration
        
        members : current vertices of interest
        
        memberData : feature data for members
    """
    
    predProbs = metaEstimator.predict_proba(memberData)
    cMaps = map(mappings.get,members)
    
    cMod = metaEstimator.classes_
    
    labels = []
    
    for i,m in enumerate(members):
        if cMaps[i]:
        
            tups = [(k,v) for k,v in enumerate(cMod) if v in cMaps[i]]
            
            inds = [t[0] for t in tups]
            vals = [v[1] for v in tups]
            
            probs = predProbs[i,:][inds]
            maxProb = list([np.argmax(probs)])
            labels.append([vals[j] for j in maxProb])
    
    return labels

def treeSoftMax(metaEstimator,mappings,members,memberData):
    
    """
    Super method for classification prediction with tree-level restriction.
    
    Parameters:
    - - - - -
        metaEstimator : single decision forest
        
        mappings : labels a vertex maps to using surface registration
        
        members : current vertices of interest
        
        memberData : feature data for members
    """
    
    cMaps = map(mappings.get,members)
    
    cMod = metaEstimator.classes_
    
    predProbs = []
    
    temporaryInds = {}.fromkeys(members)
    temporaryVals = {}.fromkeys(members)

    for i,m in enumerate(members):
        if cMaps[i]:
            tups = [(k,v) for k,v, in enumerate(cMod) if v in cMaps[i]]
            
            temporaryInds[i]=[t[0] for t in tups]
            temporaryVals[i]=[v[1] for v in tups]
            
    for estimator in metaEstimator.estimators_:
        predProbs.append(estimator.predict_proba(memberData))
        
    predLabels = []
        
    for k in np.arange(len(predProbs)):
        
        labels = []
        estimateProbs = predProbs[k]
        
        for i,m in enumerate(members):
            if cMaps[i]:
                
                inds = temporaryInds[i]
                vals = temporaryVals[i]
                
                probs = estimateProbs[i,:][inds]
                maxProb = list([np.argmax(probs)])
                labels.append([vals[j] for j in maxProb])
        
        predLabels.append(labels)
    
    predLabels = np.column_stack(predLabels)

    classification = []
        
    for i in np.arange(predLabels.shape[0]):
        
        L = list(predLabels[i,:])
        maxProb = max(set(L),key=L.count)
        classification.append(maxProb)
        
    return classification


class MultiAtlas(object):
    
    """
    Class to perform multi-atlas classification based on the combine results
    of single / multi-subject classifiers.
    
    Parameters:
    - - - - -
        feats : features to include in the models
        
        threshold : threhsold to apply to mappings
        
        scale : standardize the training data

    """
    
    def __init__(self,features,atlas_size = 1,atlases=None):
        
        """
        Method to initialize Mutli-Atlas label propagation scheme.
        
        Parameters:
        - - - - -
        
            features : features to include in each Atlas
            atlas_size : number of training subjects per atlas
            atlases : number of atlases to generate
            
        """
        
        if atlas_size < 1:
            raise ValueError('Before initializing training data, atlas_size '\
                             'must be at least 1.')
        
        if atlases is not None and atlases < 0:
            raise ValueError('atlases must be positive integer or None.')
        
        self.atlas_size = atlas_size
        self.atlases = atlases
        self.features = features
        
    def set_params(self,**kwargs):
        
        """
        Update parameters with user-specified dictionary.
        """
        
        args, varargs, varkw, defaults = inspect.getargspec(self.__init__)
        
        if kwargs:
            for key in kwargs:
                if key in args:
                    setattr(self,key,kwargs[key])

    def initializeTraining(self,trainObject):
        
        """
        Private method to load and initialize training data.
        
        Parameters:
        - - - - -
            trainObject : training data (either '.p' file, or dictionary)
                        
            **kwargs : optional arguments in MALP_INITIALIZATION
        """
        
        # can either load a single SubjectFeatures object
        # or a nested dictionary structure
        if isinstance(trainObject,str):
            trainData = ld.loadPick(trainObject)
            
            if isinstance(trainData,fd.SubjectFeatures):
                self.trainingID = trainData.ID
                trainData = cu.prepareUnitaryFeatures(trainData)
                
        # otherwise, if provided with dictionary
        elif isinstance(trainObject,dict):
            trainData = trainObject
        else:
            raise ValueError('Training object is of incorrect type.')
            
        if not trainData:
            raise ValueError('Training data cannot be empty.')
            
        subjects = trainData.keys()
        
        if not self.atlases:
            self.atlases = len(subjects)
        else:
            self.atlases = min(self.atlases,len(subjects))

        datasets = []
        
        
        if self.atlas_size == 1:
            subjectSet = np.random.choice(subjects,size=self.atlases,
                                          replace=False)

            for s in subjectSet:
                td = {s: trainData[s]}
                datasets.append(td)
        else:
            for a in np.arange(self.atlases):
                subjectSet = np.random.choice(subjects,size=self.atlas_size,
                                              replace=False)
                
                td = {s: trainData[s] for s in subjectSet}
                
                datasets.append(td)

        self.datasets = datasets
        

def parallelFitting(multiAtlas,maps,features,
                    classifier = ensemble.RandomForestClassifier(n_jobs=-1),
                    **kwargs):

    """
    Method to fit a set of Atlas objects.
    """
    
    args,_, _,_ = inspect.getargspec(classifier.__init__)
    classArgs = cu.parseKwargs(args[1:],kwargs)
    classifier.set_params(**classArgs)
    
    BaseAtlas = Atlas(feats=features)
    
    args,_,_,_ = inspect.getargspec(BaseAtlas.__init__)
    atlasArgs = cu.parseKwargs(args[1:],kwargs)
    BaseAtlas.set_params(**atlasArgs)

    # fit atlas on each component

    fittedAtlases = Parallel(n_jobs=NUM_CORES)(delayed(atlasFit)(BaseAtlas,
                            d,maps,
                            classifier=classifier) for d in multiAtlas.datasets)
    
    return fittedAtlases
    
def atlasFit(baseAtlas,data,maps,classifier,**kwargs):
    
    """
    Single model fitting step.
    """

    atl = copy.deepcopy(baseAtlas)
    
    atl.fit(data,maps,classifier = classifier)
    
    return atl


def parallelPredicting(models,testObject,testMappings,*args,**kwargs):
    
    """
    Method to predicted test labels in parallel
    """
    
    predictedLabels = Parallel(n_jobs=NUM_CORES)(delayed(atlasPredict)(models[i],
                               testObject,testMappings,
                                *args,**kwargs) for i,m in enumerate(models))
    
    return predictedLabels

def atlasPredict(model,testObject,testMappings,**kwargs):
    
    """
    Single model prediction step.
    """
    
    args,_,_,_ = inspect.getargspec(model.__init__)
    modelArgs = cu.parseKwargs(args,kwargs)
    model.set_params(**modelArgs)
    
    model.predict(testObject,testMappings)
    
    return model.predicted

    
        
    