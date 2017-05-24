#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:17:38 2017

@author: kristianeschenburg
"""
import classifier_utilities as cu
import featureData as fd
import libraries as lb
import loaded as ld

import copy
import numpy as np
import os
import pickle

from joblib import Parallel, delayed
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

from sklearn import ensemble, linear_model, multiclass
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

class Atlas(object):
    
    """
    Class to instantiate a classifier for a single training brain.
    
    Parameters:
    - - - - -
        feats : names of features to included in the models
        
        scale : standardized the training data (default == True)
        
        threshold : mapping frequency threshold of merged maps (only labels 
                    with mapping frequencies greater than threshold are included
                    in the models)
        
        **kwargs : optional arguments in ['random' : number of subjects]
        
    """
    
    def __init__(self,feats,scale=True,threshold = 0.025,
                 **kwargs):
        
        ## Check remaining features ##
        
        if not feats:
            raise ValueError('Feature list cannot be empty.')
            
        if threshold < 0 or threshold > 1:
            raise ValueError('Threshold value must be within [0,1].')
            
        if scale not in [True, False]:
            raise ValueError('Scale must be boolean.')
            
        self.features = feats
        self.scale = True
        self.threshold = threshold
            
    def _initializeTraining(self,trainObject,mergedMaps,**kwargs):
        
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
            
        # if random is set, sample subsect of training data (no replacement)
        subj = trainData.keys()
        
        if kwargs:
            if 'random' in kwargs:
                if kwargs['random'] >= 1 and kwargs['random'] <= len(subj):
                    
                    sample = np.random.choice(subj,size=kwargs['random'],
                                              replace=False)
                    sampleData = {s: trainData[s] for s in sample}
                    trainData = sampleData                    
                
        self.N = len(trainData.keys())
        print 'Models contain training size of {}.\n'.format(self.N)
        
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
            print('Warning: Training data and response vectors do not have the same keys.')
            cond = False

        if not self._compareTrainingDataSize():
            print('Warning: Training data and response vectors are not the same length.')
            cond = False
            
        if not cond:
            raise ValueError('Training data is incorrect.')
            
    def fit(self, trainObject, mergedMaps, model_type = 'ori',
            classifier = ensemble.RandomForestClassifier(n_jobs=-1),
            **kwargs):
        
        """
        Method to initialize training data and fit the classifiers.
        
        Parameters:
        - - - - -
            trainObject : training data (either '.p' file, or dictionary)
            
            mergedMaps : merged MatchingLibraries corresponding to training data
            
            model_type : type of classification scheme for multi-class 
                         A   prediction models
                            
            classifier : model
            
            **kwargs : optional arguments
            
        """
        
        if kwargs:
            swargs = {}   
            isTrue = []
            
            for key in kwargs:
                if key in ['threshold','random']:
                    isTrue.append(key)
                    swargs[key] = kwargs[key]
                    
            for key in isTrue:
                del(kwargs[key])
        
        self._initializeTraining(trainObject,mergedMaps,**swargs)
        
        self.model_type = model_type
        self.classifier = classifier
        
        if kwargs:
            kwargs = cu.parseKwargs(kwargs)
            classifier.set_params(kwargs)
            
        model_selector = {'oVo': OneVsOneClassifier(classifier),
                          'oVr': OneVsRestClassifier(classifier),
                          'ori': classifier}
        models = {}

        mgm = self.mergedMappings
        freqs = lb.mappingFrequency(mgm)
        
        thr = self.threshold
        
        
        print 'Fitting model with model_type: ' + model_type
        print 'Fitting model with threshold: ' + str(thr)+ '\n'
        
        status = np.asarray([0.25,0.5,0.75,1]); 
        c = 1

        for l in self.labels:
            if l in self.labelData.keys():
                
                # copy the model (due to passing by reference)
                models[l] = copy.deepcopy(model_selector[model_type])

                # threshold the mapping frequencies
                mapped = lb.mappingThreshold(freqs[l],thr)
                mapped.append(l)
                mapped = list(set(mapped).intersection(self.labels))
                
                # build classifier training data upon request
                [learned,y] = cu.mergeLabelData(self.labelData,self.response,mapped)
    
                models[l].fit(learned,np.squeeze(y))
                
                # to keep track of fitting procedure
                frac = 1.*c/len(self.labels)
                inds = frac >= status
                if sum(inds) > 0:
                    print 'Fitting ' + str(100*np.squeeze(status[inds])) + '% complete.'
                    status = status[~inds]
                c+=1
        
        self.models = models
        self._fit = True
        
    def _loadTest(self,y,yMatch,**kwargs):
        
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
                                
            kwargs : computing vertex-to-label mappings is time consuming, so
                    we allow the option to pre-load this data
                    allowed keys: in ['load','thresh']
        """
        
        # get Atlas attributes
        labels = self.labels
        features = self.features
        
        # analyze optional argument keys
        test_keys = ['load','save','threshold']
        
        # remove irrelevant key : value options
        for k in kwargs.keys():
            if k not in test_keys:
                del(kwargs[k])
                
        # load test subject data, save as attribtues
        testObject = ld.loadPick(y)        
        testMatch = ld.loadPick(yMatch)
        
        self.testMatch = testMatch
        self.testObject = testObject
        
        # if the traiing data has been scaled, apply scaling 
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
        
        # if label : vertex mapping file is provided, load it
        if 'load' in kwargs:
            labelToVertexMaps = ld.loadPick(kwargs['load'])
            
        # otherwise, generate it
        else:
            
            # get vertex to label mapping counts
            vTLM = copy.deepcopy(testMatch.vertLib)
            
            # convert vertex to label mapping counts to frequencies
            freqMaps = lb.mappingFrequency(vTLM)
            
            # if threshold argument is set, threshold mappings by frequency
            if 'threshold' in kwargs:
                cutoff = kwargs['threshold']
            elif self.threshold:
                cutoff = self.threshold
            else:
                cutoff = 0
                
            if cutoff > 0:
                threshed = {}.fromkeys(freqMaps.keys())
                for k in threshed.keys():
                    # if the vertex actual maps to labels (not midline)
                    if freqMaps[k]:
                        threshed[k] = lb.mappingThreshold(freqMaps[k],cutoff)
                        
                freqMaps = threshed

            self.mappingsCutoff = freqMaps
            labelToVertexMaps = cu.vertexMemberships(freqMaps,labels)
            
        self.labelToVertexMaps = labelToVertexMaps
        
        if 'save' in kwargs:
            if not os.path.isfile(kwargs['save']):
                try:
                    with open(kwargs['save'],"wb") as outFile:
                        pickle.dump(self.labelToVertexMaps,outFile,-1)
                except IOError:
                    print('Cannot save labelVertices to file.')
                    
        self._loadedTest = True
                    
    def predict(self,y,yMatch,*args,**kwargs):
        
        """
        Method to predict labels of test data.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain      
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
                    
            *args : optional arguments for prediction
                        args : ['base','forests','trees']
            
                        'BASE' : non-thresholded
                        'FORESTS' : soft-max at forest-prediction level
                        'TREES' : soft-max at tree-prediction level
                        
                    If more than 2 options are prodivided, 'base' takes precedence.
                    
            **kwargs : optional arguments for loading the test data
                        keys : ['load','save','threshold']
                        
                        'load' : loads labelVertex mappings (if exists)
                        'save' : saves labelVertex mappings (to path)
                        'threshold' : thresholds mapping frequency
                        
            
        """
        
        test_keys = ['load','save','threshold']
        prediction_values = ['BASE','FORESTS','TREES']

        # prepare optional dictionaries for loading test data and prediction
        # test labels
        if kwargs:
            twargs = {}
            for key in kwargs:
                if key in test_keys:
                    twargs[key] = kwargs[key]

        if args:
            pargs = []
            for val in args:
                if val in prediction_values:
                    pargs.append(val)
            
            if len(set(pargs)) > 1:
                pargs = list(['BASE'])

        # options for loading test data
        self.twargs = twargs
        # optios for processing test data
        self.pargs = pargs
                    
        # load the testing data
        self._loadTest(y,yMatch,**twargs)

        # get names of test data vertices
        verts = self.testMatch.vertLib.keys()
        # get test data
        mtd = self.mergedTestData

        # initialize prediction dictionary
        baseline = {k: [] for k in verts}
        
        # check to see what type of processing option was provided
        if 'BASE' in pargs or not pargs:
            
            print('{} prediction option provided.\n'.format(pargs[0]))

            for lab in self.labels:

                # compute vertices that map to that label
                members = self.labelToVertexMaps[lab]
                            
                if len(members) > 0:
    
                    # compute member labels from lab core classifier
                    scores = self._predictPoint(mtd,lab,members) 
    
                    # save results in self.predict
                    baseline = cu.updatePredictions(baseline,members,scores)
                    
        else:
            
            print('{} prediction option provided.\n'.format(pargs[0]))

            for lab in self.labels:
                members = self.labelToVertexMaps[lab]
                
                if len(members) > 0:
                    
                    if pargs[0] == 'TREES':
                        
                        predLabs = np.zeros((len(members),1))
                        
                        predLabs = treeSoftMax(self.models[lab],
                                               self.mappingsCutoff,
                                               members,
                                               mtd[members,:])
                                
                    elif pargs[0] == 'FORESTS':
                        
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

        predicted = {}
        predicted = predicted.fromkeys(storage.keys())

        for v,m in storage.items():
            if m:
                predicted[v] = max(set(m),key=m.count)
            else:
                predicted[v] = 0
                
        return predicted.values()
    
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
        
def singleTreeSoftMax(estimator,candidates_map,candidates_model,data):
    
    """
    Method to restrict individual decision tree soft-max prediction to labels 
    generated in the surface registration step.
    
    A given test vertex maps to a set of labels in the training data.  We
    expect that a classifier SHOULD produce one of these labels, but it might
    not.  As such, we constrain the soft-max prediction step to consider only
    mapped labels.
    
    Parameters:
    - - - - -
        estimator : single tree
        
        candidates_map : labels a vertex maps to using surface registration
                            
        candidates_model : candidates considered in the overarching classifier
        
        data : test data to classify
    """

    cMap = candidates_map
    cMod = candidates_model
    
    # get indices of candidate model labels if those labels are in the
    # candidate mapping labels
    tups = [(k,v) for k,v in enumerate(cMod) if v in cMap]
    
    inds = [t[0] for t in tups]
    vals = [v[1] for v in tups]
    
    # get classification probabilities associated with each of the model labels
    probs = estimator.predict_proba(data)[:,inds]
    # compute which index maximizes the classification probability
    maxProb = np.argmax(probs,axis=1)
    
    # compute label corresponding to max index
    labels = [vals[i] for i in maxProb]
    
    return labels

class MultiAtlas(object):
    
    """
    Class to perform multi-atlas classification based on the combine results
    of single / multi-subject classifiers.
    
    Parameters:
    - - - - -
        trainObject : input training data (either '.p' file, or dictionary)
        
        model : classification model to use -- spans the set of all Atlases
        
        **kwargs : optional meta-class arguments in ['splitby','full']
        **swargs : optional Atlas-class arguments in 
    """
    
    def __init__(self,feats):
        
        """
        Method to initialize Mutli-Atlas label propagation scheme.
        """
        
        self.features = feats
        
    def fit(self,trainObject,maps,**kwargs):
        
        """
        Method to fit a set of Atlas objects.
        """
        
        intialz_options = ['full','splitby']
        fit_options = ['threshold']
        
        if kwargs:
            print(kwargs)
            
            iniArgs = {}
            fitArgs = {}
            
            for k in kwargs:
                if k in intialz_options:
                    iniArgs[k] = kwargs[k]
                elif k in fit_options:
                    fitArgs[k] = kwargs[k]
            
        # intiialize component training data sets
        self._initializeTraining(trainObject,maps,self.features,**iniArgs)
        
        # fit atlas on each component
        BaseAtlas = Atlas(feats=self.features)

        fittedAtlases = Parallel(n_jobs=NUM_CORES)(delayed(self._baseFit)(BaseAtlas,
                                d,self.maps,fitArgs) for d in self.datasets)
        
        self.fittedAtlases = fittedAtlases
        
    def _baseFit(baseAtlas,data,maps,args):
        
        atl = copy.deepcopy(baseAtlas)
        mps = copy.deepcopy(maps)
        
        atl.fit(data,mps,**args)
        
        return atl

    def _initializeTraining(self,trainObject,maps,globalFeatures,**kwargs):
        
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

        if 'full' in kwargs:
            
            datasets = []
            
            subjectsSet = np.random.choice(subjects,size=kwargs['full'],replace=False)
            td = {}.fromkeys(subjectsSet)
             
            for s in subjectsSet:
                td[s] = trainData[s]
                datasets.append(td)

        self.datasets = datasets
        self.globalFeatures = globalFeatures
        self.maps = maps
        
    