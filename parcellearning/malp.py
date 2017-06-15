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
import h5py
import inspect
import os
import pickle

from joblib import Parallel, delayed
import multiprocessing

import numpy as np
from sklearn import ensemble
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# number of cores to parallelize over
NUM_CORES = multiprocessing.cpu_count()

# Valid soft-max constraints
PREDICTION_KEYS = ['BASE','TREES','FORESTS']

# type of classifier used -- the type of classifier will dictate option applicability of softmax_type
CLASSIFIER = {'random_forest': PREDICTION_KEYS,
              'decision_tree': PREDICTION_KEYS,
              'logistic': 'BASE',
              'svm': 'BASE',
              'gaussian_process': 'BASE'}

# type of label confusion data
NEIGHBORHOOD_TYPE = ['adjacency','confusion']


class Atlas(object):
    
    """
    Class to instantiate a classifier for a single training set.
    
    Parameters:
    - - - - -
        feats : names of features to included in the models
        
        scale : standardized the training data (default == True)
        
        thresh_train : threshold to apply to training subject matches (default = 0.05)
        
        thresh_test : threshold to apply to test subject matches (default = 0.05)

        hops : default neighborhood size from adjacency Dijkstra list (default = 1)

        neighborhood : type of label neighborhood to consider.  If 'adjacency', uses
                        'hop_size' and Dijkstra distance files.  If 'confusion', uses
                        'thresh_train' and MergedMappings file.
        
        softmax_type : type of classification constraining based on surface
                        registration mappings.  BASE for none, TREES at the
                        tree level, and FORESTS at the forest level.  (default = 'BASE')

        classifier_type : model used to fit training data (default = 'random_forest')

        exclude_testing : (None,List) list of subjects to exclude from the training step

        random : number of training subjects to include in model
        
        load : if test subject label-verte memberships has been generated
                previously, can provide file
                
        save : path to save label-vertex memberships

    """
    
    def __init__(self,feats,scale=True, thresh_train = 0.05, thresh_test = 0.05, hop_size = 1, neighborhood = 'adjacency',
                 softmax_type = 'BASE', classifier_type = 'random_forest', exclude_testing = None, random = None,
                 load = None, save = None):

        # check feature value
        if not feats and not isinstance(feats,list):
            raise ValueError('Features cannot be empty.  Must be a list.')

        # check scale type value
        if scale not in [True, False]:
            raise ValueError('Scale must be boolean.')

        # check thresh_train value
        if thresh_train < 0 or thresh_train > 1:
            raise ValueError('Threshold value must be within [0,1].')

        # check tresh_test value
        if thresh_test < 0 or thresh_test > 1:
            raise ValueError('Threshold value must be within [0,1].')

        # check hop_size value
        if hop_size < 0 and not isinstance(hop_size,int):
            raise ValueError('hop size must be non-negative integer value.')

        if neighborhood not in NEIGHBORHOOD_TYPE:
            raise ValueError('neighbohood must be in {}.'.format(' '.join(NEIGHBORHOOD_TYPE)))

        # check softmax constraint value
        if softmax_type not in PREDICTION_KEYS:
            raise ValueError('softmax_type must be in {}.'.format(' '.join(PREDICTION_KEYS)))

        # check classifier_type value
        if classifier_type not in CLASSIFIER.keys():
            raise ValueError('classifier_type must be in {}.'.format(' '.join(CLASSIFIER.keys())))

        # check exclude_testing value
        if exclude_testing is not None and not isinstance(exclude_testing,str) and not isinstance(exclude_testing,list):
            raise ValueError('exclude_testing must by a string or None.')

        # check random value
        if random is not None and random < 0:
            raise ValueError('Random must be a positive integer or None.')

        # check load value
        if not load is None and not isinstance(load,str):
            raise ValueError('load must be a string or None.')

        # check save value
        if save is not None and not isinstance(save,str):
            raise ValueError('save must be a string or None.')

        # CLASSIFIER maps classifier_type to valid softmax constraint types
        # softmax_type only applies to 'random_forest' or 'decision_tree'
        if softmax_type not in CLASSIFIER[classifier_type]:
            softmax_type = 'BASE'

        # training-related attributes
        self.features = feats
        self.scale = scale
        self.thresh_train = thresh_train
        self.hop_size = hop_size
        self.neighborhood = neighborhood
        self.exclude_testing = exclude_testing

        # testing-related attributes
        self.thresh_test = thresh_test
        self.softmax_type = softmax_type
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
            
    def fit(self, trainObject, neighborhoodMap, model_type = 'ori',
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

        self.model_type = model_type
        self._initializeTraining(trainObject,neighborhoodMap)

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

        neighbors = self.neighbors

        for i,l in enumerate(self.labels):
            if l in self.labelData.keys() and l in neighbors.keys():

                # copy the model (due to passing by reference)
                models[l] = copy.deepcopy(model_selector[model_type])

                # get associated neighboring regions (from adjacency or confusion data)
                label_neighbors = neighbors[l]
                label_neighbors.append(l)

                label_neighbors = list(set(label_neighbors).intersection(self.labels))
                
                # build classifier training data upon request
                [learned,y] = cu.mergeLabelData(self.labelData,self.response,label_neighbors)
    
                models[l].fit(learned,np.squeeze(y))

        self.models = models
        self._fit = True
        
    def _initializeTraining(self,trainObject,neighborhoodMap):
        
        """
        Initialize the object with the training data.
        
        Parameters:
        - - - - -
            trainObject : input training data (either '.p' file, or dictionary)
            
            neighborhoodMap : Dijkstra distance file or MergedMappings file

        """

        # load the training data

        if isinstance(trainObject,str):
            trainData = ld.loadH5(trainObject,*['full'])
        elif isinstance(trainObject,h5py._hl.files.File):
            trainData = trainObject
        elif isinstance(trainObject,dict):
            trainData = trainObject
        else:
            raise ValueError('Training object is of incorrect type.')

        if not trainData:
            raise ValueError('Training data cannot be empty.')

        if isinstance(trainData,h5py._hl.files.File):
            parseFeatures = copy.deepcopy(self.features)
            parseFeatures.append('label')

            parsedData = ld.parseH5(trainData,parseFeatures)
            trainData.close()
            trainData = parsedData

        # get subject IDs in training data
        subjects = trainData.keys()

        # if exclude_testing is set, the data for these subjects when fitting the models
        if self.exclude_testing:
            subjects = list(set(subjects).difference(set(self.exclude_testing)))
        
        # if random is set, select random subset of size random from viable training subjects
        if not self.random:
            randomSample = len(subjects)
        else:
            randomSample = min(self.random,len(subjects))

        sample = np.random.choice(subjects,size=randomSample,replace=False)
        sampleData = {s: trainData[s] for s in sample}
        trainData = sampleData

        # if scale is True, scale each feature of the training data and save the transformation
        # transformation will be applied to incoming test data
        if self.scale:
            [trainData,scalers] = cu.standardize(trainData,self.features)
            
            # scaler objects to transform test data
            self.scalers = scalers
            self._scaled = True

        # get unique labels in training set
        self.labels = set(cu.getLabels(trainData)).difference({0,-1})
        
        # isolate training data corresponding to each label
        self.labelData = cu.partitionData(trainData,feats = self.features)
        
        # build response vector for each label
        self.response = cu.buildResponseVector(self.labels,self.labelData)

        # load and prepare neighborhoodMap
        neighborhoodMap = ld.loadPick(neighborhoodMap)

        if self.neighborhood == 'adjacency':
            boundary = 'inside'
            threshold = self.hop_size
            
        else:
            boundary = 'outside'
            threshold = self.thresh_train

            neighborhoodMap = lb.mappingFrequency(neighborhoodMap)
            neighborhoodMap = lb.mappingThreshold(neighborhoodMap, threshold, boundary)
            
        self.neighbors = neighborhoodMap

        # check quality of training data to ensure all features have same length,
        # all response vectors have the same number of samples, and that all training data
        # has the same features
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
                if lab in self.neighbors.keys():

                    # compute vertices that map to that label
                    members = self.labelToVertexMaps[lab]
                                
                    if len(members) > 0:
        
                        # compute member labels from lab core classifier
                        scores = self._predictPoint(mtd,lab,members) 
        
                        # save results in self.predict
                        baseline = cu.updatePredictions(baseline,members,scores)
                    
        else:
            for lab in self.labels:
                if lab in self.neighbors.keys():
                    
                    members = self.labelToVertexMaps[lab]
                    
                    if len(members) > 0:
                        
                        if softmax_type == 'TREES':
                            
                            sfmxLabs = treeSoftMax(self.models[lab],
                                                   self.mappingsCutoff,
                                                   members,
                                                   mtd[members,:])
                                    
                        elif softmax_type == 'FORESTS':
                            
                            sfmxLabs = forestSoftMax(self.models[lab],
                                                            self.mappingsCutoff,
                                                            members,
                                                            mtd[members,:])
    
                        predLabs = np.squeeze(sfmxLabs)
                        baseline = cu.updatePredictions(baseline,members,
                                                        predLabs)

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
        testObject = ld.loadH5(y,*['full'])
        ID = testObject.attrs['ID']
        
        parsedData = ld.parseH5(testObject,self.features)
        testObject.close()
        testObject = parsedData
        
        data = testObject[ID]
        
        testMatch = ld.loadPick(yMatch)

        self.testMatch = testMatch
        self.testObject = testObject

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
        # and threshold labels to include by the mapping frequency
        freqMaps = lb.mappingFrequency(vTLM)
        threshed = lb.mappingThreshold(freqMaps,threshold,'outside')
        self.mappingsCutoff = threshed
        
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
            if len(probs):
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
                if len(probs):
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
    
    def __init__(self,features,atlas_size = 1,atlases=None,
                 exclude_testing = None,cv=False,cv_size=20):
        
        """
        Method to initialize Mutli-Atlas label propagation scheme.
        
        Parameters:
        - - - - -
        
            features : features to include in each Atlas

            atlas_size : number of training subjects per atlas

            atlases : number of atlases to generate

            exclude_testing = (None,str,list) list of subjects to exclude from training data
            
        """
        
        if atlas_size < 1:
            raise ValueError('Before initializing training data, atlas_size '\
                             'must be at least 1.')
        
        if atlases is not None and atlases < 0:
            raise ValueError('atlases must be positive integer or None.')

        if exclude_testing is not None and not isinstance(exclude_testing,str) and not \
                isinstance(exclude_testing,list):
            raise ValueError('exclude_testing must by a string or None.')

        if not isinstance(cv,bool):
            raise ValueError('cv must be of type boolean.')

        if not isinstance(cv_size,int) and cv_size < 1:
            raise ValueError('cv_size must be a positive integer.')
        
        self.atlas_size = atlas_size
        self.atlases = atlases
        self.features = features
        self.exclude_testing = exclude_testing

        self.cv = cv
        self.cv_size = cv_size
        
    def set_params(self,**kwargs):
        
        """
        Update parameters with user-specified dictionary.
        """
        
        try: 
            self.initialized
        except:
            args, varargs, varkw, defaults = inspect.getargspec(self.__init__)
            
            if kwargs:
                for key in kwargs:
                    if key in args:
                        setattr(self,key,kwargs[key])
        else:
            print('Warning: training data has already been intiailized.  '
                  'New parameters will have no effect.  '
                  'Reinstantiate the class to update the parameters.')

    def initializeTraining(self,trainObject,**kwargs):
        
        """
        Private method to load and initialize training data.
        
        Parameters:
        - - - - -
            trainObject : training data (either '.p' file, or dictionary)
        """
        
        # load a single SubjectFeatures object or a nested dictionary structure
        if isinstance(trainObject,str):
            trainData = ld.loadH5(trainObject,*['full'])
        elif isinstance(trainObject,h5py._hl.files.File):
            trainData = trainObject
        elif isinstance(trainObject,dict):
            trainData = trainObject
        else:
            raise ValueError('Training object is of incorrect type.')
            
        if not trainData:
            raise ValueError('Training data cannot be empty.')

        # pre-process if trainData is h5py
        if isinstance(trainData,h5py._hl.files.File):
            parseFeatures = copy.deepcopy(self.features)
            parseFeatures.append('label')

            parsedData = ld.parseH5(trainData,parseFeatures)
            trainData.close()
            trainData = parsedData
            
        subjects = trainData.keys()

        if kwargs:
            self.set_params(**kwargs)

        if self.exclude_testing:
            print(self.exclude_testing)
            print(type(self.exclude_testing))
            subjects = list(set(subjects).difference(set(self.exclude_testing)))
        
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
        self.initialized=True
        

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

