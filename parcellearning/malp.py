#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:17:38 2017

@author: kristianeschenburg

"""

import classifier_utilities as cu
import loaded as ld
import matchingLibraries as lb

import copy
import h5py
import inspect
import os
import pickle

from joblib import Parallel, delayed
import multiprocessing

import numpy as np
from sklearn import ensemble,preprocessing
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
    
    def __init__(self,scale=True, thresh_train = 0.05, hop_size = 1, neighborhood = 'adjacency',
                 softmax_type = 'BASE', classifier_type = 'random_forest', 
                 exclude_testing = None, random = None,load = None, save = None):

        # check scale type value
        if scale not in [True, False]:
            raise ValueError('Scale must be boolean.')

        # check thresh_train value
        if thresh_train < 0 or thresh_train > 1:
            raise ValueError('Threshold value must be within [0,1].')

        # check hop_size value
        if hop_size < 0 and not isinstance(hop_size,int):
            raise ValueError('hop size must be non-negative integer value.')

        if neighborhood not in NEIGHBORHOOD_TYPE:
            raise ValueError('neighbohood must be in {}.'.format(' '.join(NEIGHBORHOOD_TYPE)))

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

        # training-related attributes
        self.scale = scale
        self.thresh_train = thresh_train
        self.hop_size = hop_size
        self.neighborhood = neighborhood
        self.exclude_testing = exclude_testing

        # testing-related attributes
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
            
    def fit(self, x_train, y_train, model_type='ori',
            classifier = ensemble.RandomForestClassifier(n_jobs=-1),**kwargs):
        
        """
        Method to initialize training data and fit the classifiers.
        
        Parameters:
        - - - - -

            mergedMaps : merged MatchingLibraries corresponding to training data
            
            model_type : type of classification scheme for multi-class 
                         A   prediction models
                            
            classifier : model to use

            kwargs : optional arguments for classifier
        """

        self.model_type = model_type
        labelData = x_train
        response = y_train

        labels = self.labels

        # get valid arguments for supplied classifier
        # get valid parameters passed by user
        # update classifier parameters
        # save base models
        classifier_params = inspect.getargspec(classifier.__init__)
        classArgs = cu.parseKwargs(classifier_params,kwargs)
        classifier.set_params(**classArgs)
        self.classifier = classifier
        
        print 'depth: {}'.format(classifier.max_depth)
        print 'nEst: {}'.format(classifier.n_estimators)
            
        model_selector = {'oVo': OneVsOneClassifier(classifier),
                          'oVr': OneVsRestClassifier(classifier),
                          'ori': classifier}
        models = {}

        neighbors = self.neighbors

        for i,l in enumerate(self.labels):
            if l in labelData.keys() and l in neighbors.keys():

                # copy the model (due to passing by reference)
                models[l] = copy.deepcopy(model_selector[model_type])

                # get associated neighboring regions (from adjacency or confusion data)
                labelNeighbors = neighbors[l]
                labelNeighbors.append(l)

                labelNeighbors = list(set(labelNeighbors).intersection(labels))
                
                # build classifier training data upon request
                [learned,y] = cu.mergeLabelData(labelData,response,labelNeighbors)
    
                models[l].fit(learned,np.squeeze(y))

        self.models = models
        self._fit = True

    def loadTraining(self,trainObject,neighborhoodMap,dataDir,hemisphere,
                     features):
        
        """
        Initialize the object with the training data.
        
        Parameters:
        - - - - -
            trainObject : input training data (either '.p' file, or dictionary)
            
            neighborhoodMap : Dijkstra distance file or MergedMappings file
            
            dataDir : input directory where data exists
            
            hemisphere : 'Left' or 'Right'
            
            features : features to train the classifier on ('label' is implicit)

        """

        # check feature value
        if not features and not isinstance(features,list):
            raise ValueError('Features cannot be empty.  Must be a list.')
        else:
            self.features = features

        # load the training data
        loadingFeatures = copy.copy(features)
        loadingFeatures.append('label')

        if isinstance(trainObject,str):
            trainData = ld.loadH5(trainObject,*['full'])
        elif isinstance(trainObject,h5py._hl.files.File) or isinstance(trainObject,dict):
            trainData = trainObject
        elif isinstance(trainObject,list):
            trainData = loadDataFromList(trainObject,dataDir,loadingFeatures,hemisphere)
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
        trainData = {s: trainData[s] for s in sample}
        
        
        
        training = []
        labels = []
        
        trainFeatures = list(set(self.features).difference('label'))
        for subj in trainData.keys():
            training.append(cu.mergeFeatures(trainData[subj],trainFeatures))
            labels.append(cu.mergeFeatures(trainData[subj],['label']))
        
        trainData = np.squeeze(np.row_stack(training))
        labelVector = np.squeeze(np.concatenate(labels))
        self.labels = set(labelVector).difference({0,-1})
        
        
        if self.scale:
            
            scaler = preprocessing.StandardScaler(with_mean=True,with_std=True)
            trainData = scaler.fit_transform(trainData)
            self.scaler = scaler
            self.scaled=True


        # isolate training data corresponding to each label
        labelData = cu.partitionData(trainData,labelVector,self.labels)
        response = cu.buildResponseVector(self.labels,labelData)
        
        self.input_dim = labelData[labelData.keys()[0]].shape[1]

        # load and prepare neighborhoodMap
        neighborhoodMap = ld.loadPick(neighborhoodMap)

        if self.neighborhood == 'adjacency':
            boundary = 'inside'
            threshold = self.hop_size
            
        else:
            boundary = 'outside'
            threshold = self.thresh_train

            neighborhoodMap = lb.mappingFrequency(neighborhoodMap)
        
        self.neighbors = lb.mappingThreshold(neighborhoodMap, threshold, boundary)



        # check quality of training data to ensure all features have same length,
        # all response vectors have the same number of samples, and that all training data
        # has the same features
        cond = True
        if not compareTrainingDataKeys(labelData,response):
            print('WARNING: Label data and label response do not have same keys.')
            cond = False

        if not compareTrainingDataSize(labelData,response):
            print('WARNING: Label data and label response are not same shape.')
            cond = False

        if not cond:
            raise ValueError('Training data is flawed.')
            
        return [labelData,response]


    def predict(self,y,yMatch, yMids, softmax_type = 'BASE'):
        
        """
        Method to predict labels of test data.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain      
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
                    
        """

        funcs = {'BASE': baseSoftMax,
                 'TREES': treeSoftMax,
                 'FORESTS': forestSoftMax}
        
        labels = self.labels
        neighbors = self.neighbors

        # Python is 0-indexed, while Matlab is not
        # We adjust the Matlab coordinates by subtracting 1
        midline = ld.loadMat(yMids)-1

        R = 180
        # For now, we provide the paths to predict
        # In the future it might make sense to provide the data arrays
        [mm,mtd,ltvm] = self.loadTest(y,yMatch)

        mm[midline,:] = 0
        mtd[midline,:] = 0
        
        [xTest,yTest] = mtd.shape
        if yTest != self.input_dim:
            raise Warning('Test data does not have the same number \
                          features as the training data.')

        # initialize prediction dictionary
        baseline = np.zeros((mtd.shape[0],R+1))

        for lab in labels:
            if lab in neighbors.keys():
                
                members = ltvm[lab]
                memberData = mtd[members,:]
                estimator = self.models[lab]
                
                if len(members) > 0:
                    preds = funcs[softmax_type](estimator,members,memberData,mm,R)
                    baseline = cu.updatePredictions(baseline,members,preds)
                
        predicted = np.argmax(baseline,axis=1)
        self.baseline = baseline
        self.predicted = predicted

        return (baseline,predicted)

    def loadTest(self,y,yMatch):
        
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
        load = self.load
        save = self.save
        
        features = self.features
 
        # load test subject data, save as attribtues
        tObject = ld.loadH5(y,*['full'])
        ID = tObject.attrs['ID']
        
        parsedData = ld.parseH5(tObject,features)
        tObject.close()

        data = parsedData[ID]
        mtd = cu.mergeFeatures(data,features)
        print 'Testing shape: {}'.format(mtd.shape)

        if self.scaled:
            scaler = self.scaler
            mtd = scaler.transform(mtd)
            
        threshed = ld.loadMat(yMatch)

        # Computing label-vertex memberships is time consuming
        # If already precomputed for given test data at specified threshold,
        # can supply path to load file.
        if load:
            if os.path.isfile(load):
                ltvm = ld.loadPick(load)
        # Otherwise, compute label-vertex memberships.
        else:
            ltvm = cu.vertexMemberships(threshed,180)
        
        self.ltvm = ltvm

        # if save is provided, save label-vertex memberships to file
        if save:
            try:
                with open(save,"wb") as outFile:
                    pickle.dump(self.labelToVertexMaps,outFile,-1)
            except IOError:
                print('Cannot save label-vertex memberships to file.')
                
        return [threshed,mtd,ltvm]

def compareTrainingDataSize(labelData,response):
    
    """
    Method to ensure that the length of the response vector is the same 
    length as the number of observations in the training feature data.
    
    This must be true in order to actually train the classifiers for each
    label.
    """
    cond = True

    for f,r in zip(set(labelData.keys()),set(response.keys())):
        
        sf = labelData[f].shape[0]
        sr = response[r].shape[0]
        
        if sf != sr:
            cond = False
    
    return cond
        
def compareTrainingDataKeys(labelData,response):
    
    """
    Method to ensure that the keys for the training data for the response
    vectors are the same.  These must be the same in order to properly
    access the training data for training the classifiers.
    """

    sf = set(labelData.keys())
    sr = set(response.keys())
    
    return sf == sr


def baseSoftMax(metaEstimator,members,memberData,mm,R):
    
    """
    """

    print 'Base Prediction'
    
    predicted = np.squeeze(metaEstimator.predict(memberData))

    return predicted
        
def forestSoftMax(metaEstimator,members,memberData,mm,R):
    
    """
    Method to restrict whole random decision forest soft-max prediction to
    labels generated in the surface registration step.
    
    Parameters:
    - - - - -
        
        metaEstimator : single decision forest
                
        members : current vertices of interest
        
        memberData : feature data for members
        
        mm : binary matching matrix
        
        R : number of labels in training set
    """

    memberMatrix = mm[members,:]
    predProbs = metaEstimator.predict_proba(memberData)
    
    classes = metaEstimator.classes_
    forestMatrix = np.zeros((len(members),mm.shape[1]+1))
    forestMatrix[:,classes] = predProbs
    forestMatrix = forestMatrix[:,1:]
    
    predThresh = memberMatrix*forestMatrix
    
    labels = np.argmax(predThresh,axis=1)+1

    return labels

def treeSoftMax(metaEstimator,members,memberData,mm,R):
    
    """
    Super method for classification prediction with tree-level restriction.
    
    Parameters:
    - - - - -
        metaEstimator : single decision forest
                
        members : current vertices of interest
        
        memberData : feature data for members
        
        mm : binary matching matrix
        
        R : number of labels in training set
        
    A single metaEstimator consists of a set of sub-estimators.  The classes 
    for the metaEstimator corresponds to a list of K values to predict 
    i.e. [1,5,6,7,100]
    
    Each sub-estimator, however, has 0-K indexed classes (i.e. [0,1,2,3,4]), 
    where the index corresponds to the position in the metaEstimator 
    class list.
    """

    print 'Tree Prediction'

    memberMatrix = mm[members,:]

    treeProbs = []
    classes = []
    for est in metaEstimator.estimators_:
        classes.append(metaEstimator.classes_)
        treeProbs.append(est.predict_proba(memberData))

    predictedLabels = []
    
    treeMatrix = np.zeros((len(members),mm.shape[1]+1))
        
    for k in np.arange(len(treeProbs)):
        
        treeClasses = np.squeeze(classes[k]).astype(np.int32)
        treeProbabilities = treeProbs[k]
        
        tm = copy.deepcopy(treeMatrix)
        tm[:,treeClasses] = treeProbabilities
        tm = tm[:,1:]
        
        treeThresh = memberMatrix*tm
        predictedLabels.append(np.argmax(treeThresh,axis=1)+1)
    
    predictedLabels = np.column_stack(predictedLabels)

    classification = []
        
    for i in np.arange(predictedLabels.shape[0]):
        
        L = list(predictedLabels[i,:])
        maxProb = max(set(L),key=L.count)
        classification.append(maxProb)
    
    classification = np.asarray(classification)
        
    return classification

def loadDataFromList(subjectList,dataDir,features,hemi):
    
    """
    Generates the training data for the neural network.
    
    Parameters:
    - - - - -
        subjectList : list of subjects to include in training set
        dataDir : main directory where data exists -- individual features
                    will exist in sub-directories here
        features : list of features to include
        hemi : hemisphere to process
    """
    
    hemisphere = {}.fromkeys('Left','Right')
    hemisphere['Left'] = 'L'
    hemisphere['Right'] = 'R'
    
    H = hemisphere[hemi]
    
    # For now, we hardcode where the data is
    trainDir = '{}TrainingObjects/FreeSurfer/'.format(dataDir)
    trainExt = '.{}.TrainingObject.aparc.a2009s.h5'.format(H)
    
    midDir = '{}Midlines/'.format(dataDir)
    midExt = '.{}.Midline_Indices.mat'.format(H)

    data = {}

    for s in subjectList:

        # Training data
        trainObject = '{}{}{}'.format(trainDir,s,trainExt)
        midObject = '{}{}{}'.format(midDir,s,midExt)

        # Check to make sure all 3 files exist
        if os.path.isfile(trainObject) and os.path.isfile(midObject):

            # Load midline indices
            # Subtract 1 for differece between Matlab and Python indexing
            mids = ld.loadMat(midObject)-1
            mids = set(mids)

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
            
            data[s] = subjData[s]

    return data


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
    
    def __init__(self,atlas_size = 1,atlases=None,
                 exclude_testing = None):
        
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

        self.atlas_size = atlas_size
        self.atlases = atlases
        self.exclude_testing = exclude_testing

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

    def loadTraining(self,trainObject,dataDir,hemisphere,features,**kwargs):
        
        """
        Private method to load and initialize training data.
        
        Parameters:
        - - - - -
            trainObject : training data (either '.p' file, or dictionary)
        """
        
        loadingFeatures = copy.copy(features)
        loadingFeatures.append('label')
        
        # load a single SubjectFeatures object or a nested dictionary structure
        if isinstance(trainObject,str):
            trainData = ld.loadH5(trainObject,*['full'])
        elif isinstance(trainObject,h5py._hl.files.File):
            trainData = trainObject
        elif isinstance(trainObject,dict):
            trainData = trainObject
        elif isinstance(trainObject,list):
            trainData = loadDataFromList(trainObject,dataDir,loadingFeatures,hemisphere)
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
            subjects = list(set(subjects).difference(set(self.exclude_testing)))
        
        if not self.atlases:
            self.atlases = len(subjects)
        else:
            self.atlases = min(self.atlases,len(subjects))

        datasets = []

        # If the number of atlases is 1, all subject data will be aggregated
        # into a single dictionary
        if self.atlas_size == 1:
            subjectSet = np.random.choice(subjects,size=self.atlases,
                                          replace=False)

            for s in subjectSet:
                td = {s: trainData[s]}
                datasets.append(td)
        
        # Otherwise, N = self.atlases will be created, each with 
        # K = self.atlas_size (possibly overlapping) training subjects in it
        # This will be N separate training dictionaries datasets.
        else:
            size = self.atlas_size
            chunks = len(subjects) / size
            np.random.shuffle(subjects)
            
            # Make self.atlases independent atlases
            if chunks >= self.atlases:                
                for j in np.arange(self.atlases):
                    subjectSet = subjects[j*size:(j+1)*size]
                    td = {s: trainData[s] for s in subjectSet}
                    datasets.append(td)
            # Make as many independent atlases as possible, then select subjects
            # randomly until self.atlases have been created
            else:
                rem = self.atlases-chunks
                for j in np.arange(chunks):
                    subjectSet = subjects[j*size:(j+1)*size]
                    td = {s: trainData[s] for s in subjectSet}
                    datasets.append(td)
                for j in np.arange(rem):
                    subjectSet = np.random.choice(subjects,size=size,replace=False)
                    td = {s: trainData[s] for s in subjectSet}
                    datasets.append(td)

        self.datasets = datasets
        self.initialized=True
        

def parallelFitting(multiAtlas,maps,features,
                    classifier = ensemble.RandomForestClassifier(n_jobs=-1),
                    **kwargs):

    """
    Method to fit a set of Atlas objects.
    
    Parameters:
    - - - - -
        multiAtlas : object containing independent datasets
        maps : label neighborhood map
        features : features to include in model
    """
    
    args,_, _,_ = inspect.getargspec(classifier.__init__)
    classArgs = cu.parseKwargs(args[1:],kwargs)
    classifier.set_params(**classArgs)
    
    print 'Classifier depth: {}'.format(classifier.__dict__['max_depth'])
    print 'Classifier nEst: {}'.format(classifier.__dict__['n_estimators'])
    
    BaseAtlas = Atlas()
    
    args,_,_,_ = inspect.getargspec(BaseAtlas.__init__)
    atlasArgs = cu.parseKwargs(args[1:],kwargs)
    BaseAtlas.set_params(**atlasArgs)

    # fit atlas on each componentraransarra

    fittedAtlases = Parallel(n_jobs=NUM_CORES)(delayed(atlasFit)(BaseAtlas,
                            d,maps,features,classifier=classifier,
                            **kwargs) for d in multiAtlas.datasets)
    
    return fittedAtlases
    
def atlasFit(baseAtlas,data,maps,features,classifier,**kwargs):
    
    """
    Single model fitting step.
    """
    
    tr,re = baseAtlas.loadTraining(data,maps,None,None,features)

    atl = copy.deepcopy(baseAtlas)
    atl.fit(tr,re,classifier = classifier)
    
    return atl


def parallelPredicting(models,testObject,testMatch,testMids,**kwargs):
    
    """
    Method to predicted test labels in parallel
    """
    
    predictedLabels = Parallel(n_jobs=NUM_CORES)(delayed(atlasPredict)(models[i],
                               testObject,testMatch,testMids,
                               softmax_type='FORESTS') for i,m in enumerate(models))

    predictedLabels = np.column_stack(predictedLabels)

    classification = []
        
    for i in np.arange(predictedLabels.shape[0]):
        
        L = list(predictedLabels[i,:])
        maxProb = max(set(L),key=L.count)
        classification.append(maxProb)
    
    classification = np.asarray(classification)
    
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

