# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:25:55 2017

@author: kristianeschenburg
"""

from copy import deepcopy

import classifier_utilities as cu
import matchingLibraries as lb
import loaded as ld

import copy
import h5py
import inspect
import numpy as np
import os
import pickle

from sklearn import covariance,mixture,neighbors,preprocessing

##########
##########
##########
##########

class GMM(object):
    
    """
    Gaussian Mixture Model class to classify a given test data point by 
    modeling the unique feature profiles of a given region as originating 
    from a Gaussian Mixture Model.
    
    Parameters:
    - - - - - -
    
        trainObj : GroupFeatures object containing training data and features 
                    of interest
                    
        feats : names of features to use in classification
        
        scale : boolean, scale the training data and apply to test data
            
    """
    
    def __init__(self,scale=True,exclude_testing=None,
                 random=None,load=None,save=None,power=None):

        if not isinstance(scale,bool):
            raise ValueError('Scale must be boolean.')

        if exclude_testing is not None and not isinstance(exclude_testing,str):
            raise ValueError('exclude_testing must by a string or None.')
            
        if random is not None and random < 0:
            raise ValueError('Random must be a positive integer or None.')
            
        if not load is None and not isinstance(load,str):
            raise ValueError('load must be a string or None.')
            
        if save is not None and not isinstance(save,str):
            raise ValueError('save must be a string or None.')
            
        if power is not None and not isinstance(power,float):
            raise ValueError('power must be a float or None.')
        
        self.scale = scale
        self.exclude_testing = exclude_testing
        self.random = random
        self.load = load
        self.save = save
        self.power = power
        
    def set_params(self,**kwargs):
        
        """
        Update parameters with user-specified dictionary.
        """
        
        args, varargs, varkw, defaults = inspect.getargspec(self.__init__)
        
        if kwargs:
            for key in kwargs:
                if key in args:
                    setattr(self,key,kwargs[key])

    def fit(self, x_train,
            model = mixture.GaussianMixture(n_components=2,covariance_type='diag'),
            **kwargs):
        
        """
        Method to model the data for each training label as a mixture of
        components.  For now, we will us GaussianMixtureModels as specify the 
        number of components in each parcel.
        
        Parameters:
        - - - - - 
            trainObject : input training data (either '.p' file, or dictionary)
                
            model : dictionary of options that will be supplied to
                        sklearn.mixture.GaussianMixture()
                        
                        See sklearn documentation for more details.
                        
            kwargs : optional arguments with which to update the model
        """
        
        self.model = model
        labelData = x_train

        labels = self.labels

        args,_,_,_ = inspect.getargspec(model.__init__)
        
        modelArgs = cu.parseKwargs(args,kwargs)
        model.set_params(**modelArgs)
        
        print 'covariance type: {}'.format(model.covariance_type)
        print 'n components type: {}'.format(model.n_components)

        mixtures = {}.fromkeys(self.labels)

        for lab in labels:
            
            mixtures[lab] = deepcopy(model)
            mixtures[lab].fit(labelData[lab])

        self.mixtures = mixtures
        self._fitted = True
        
        
    def loadTraining(self,trainObject,dataDir,hemisphere,features):
        
        """
        Parameters:
        - - - - -
            trainObject : input training data (either '.p' file, or dictionary)
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
        
        #trainFeatures = list(set(self.features).difference({'label'}))
        
        print 'Model features: {}'.format(features)
        
        nf = []
        for f in self.features:
            if f != 'label':
                nf.append(f)
        
        for subj in trainData.keys():
            training.append(cu.mergeFeatures(trainData[subj],nf))
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


    def loadTest(self,y,yMatch):
        
        """
        Method to load the test data into the object.  We might be interested
        in loading new test data, so we have explicitly defined this is
        as a method.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain      
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data

        """

        # load test subject data, save as attribtues
        tObject = ld.loadH5(y,*['full'])
        ID = tObject.attrs['ID']
        
        parsedData = ld.parseH5(tObject,self.features)
        tObject.close()

        data = parsedData[ID]
        mtd = cu.mergeFeatures(data,self.features)
        print 'Testing shape: {}'.format(mtd.shape)

        if self.scaled:
            scaler = self.scaler
            mtd = scaler.transform(mtd)
            
        threshed = ld.loadMat(yMatch)

        ltvm = cu.vertexMemberships(threshed,180)

        return [threshed,mtd,ltvm]


    def predict(self,mtd,testLTVM):
        
        """
        Method to compute Mahalanobis distance of test data from the
        distribution of all training data for each label.
        
        Parameters:
        - - - - - 
        
        **kwargs : if power parameter is defined in kwargs, will perform
                    base classification and weighted classification of the
                    surface vertices
        """

        ltvm = testLTVM
        
        R = 180
        labels = self.labels

        xTest,yTest = mtd.shape
        if yTest != self.input_dim:
            raise Warning('Test data does not have the same number \
                          features as the training data.')

        # initialize prediction dictlionary
        baseline = np.zeros((mtd.shape[0],R+1))

        # for all labels in in training set
        for lab in labels:
            # compute vertices that map to that label
            members = ltvm[lab]
            memberData = mtd[members,:]
            estimator = self.mixtures[lab]
                        
            if len(members) > 0:
                
                scores = estimator.score_samples(memberData)

                # save results in self.predict
                baseline[members,lab] = scores
                
        predicted = np.argmin(baseline,axis=1)
        
        self.baseline = baseline
        self.predicted = predicted
        self._classified = True
        
        """
        if self.power:
            weightedLL = self.weight(baseline,self.power)
            self.weighted = self._classify(weightedLL)
        """
    
    def weight(self,baseline,power):
        
        """
        Method to weight the score by the frequency with which
        the test vertex mapped to the training labels.
        
        Parameters:
        - - - - -
        
            base : likehoods computed without mapping frequency
            
            power : mapping frequency exponent
            
        """
        
        vertLib = self.testMatch

        weighted = {}
        weighted = weighted.fromkeys(vertLib.keys())
    
        # for each vertex in test brain, and its mapped labels
        for vert,mapped in vertLib.items():
            vert = np.int(vert)
            
            # if vertex is mapped to labels
            if mapped:

                # get log-likelihood for each label mixture model a vertex 
                # is mapped to
                logScore = baseline[vert]

                # return weighted log-likelihoods
                weighted[vert] = self._weightPoint(mapped,logScore,power)
                
        return weighted
            
    def _weightPoint(self,mapLabels,labelScore,power):
        
        """
        Method to weight the scores of a single vertex.  Since we want the 
        maximum log-liklihood, weighting by the frequency will upweight labels 
        that are mapped to more frequently.
        
        If weight != 0, we weight the distances by 
            
            (mapCount / sumCounts)^power
        
        Parameters:
        - - - - -
        
            mapLabels : dictionary of labels to which the test vertex is mapped
                        and frequency with which those labels are mapped to
                        
            labelScore : dictionary of labels to which a test vertex is mapped
                        and log-likelihood between vertex test point given 
                        that labels mixture model
                        
            power : power to which the raise the mapping frequency to
                        
        Returns:
        - - - -
            
            weight : dictionary of weighted log-likelihoods for each of 
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
                # apply weight to log-likelihood score
                weight[l] = lWeight*labelScore[l]
                
        return weight
    
    
    def assignComponent(self,predicted):
        
        """
        Method to assign test data points to a component within the label
        they were assigned to.
        
        Parameters:
        - - - - -
        
            predicted : classified label vector
        """
        
        def mapComponent(n,p,x):
            
            """
            Defines mapping of predicted labels to component labels.
            
            Parameters:
            - - - - -
            
                n : number of components
                p : label value
                x : assignment vector
            """
            
            return (p-1)*n + (x+1)
        
        n = self._model.n_components
        
        class_vertices = self.findClassVertices(predicted)
        
        assigned = np.zeros(shape=(len(predicted),1))
        
        for lab in self._labels:
            
            members = class_vertices[lab]
            memberData = self.mergedData[members,:]
            
            if len(memberData):
                
                memberComponent = self.mixtures[lab].predict(memberData)
                print(set(memberComponent))
                assigned[members,0] = mapComponent(n,lab,memberComponent)
        
        assigned = np.squeeze(assigned)
        
        return assigned
        
    def findClassVertices(self,predicted):
        
        """
        Method to compute the which vertex is assigned to each label.
        """
        
        class_vertices = {}
        class_vertices = class_vertices.fromkeys(self._labels)
        
        for lab in self._labels:
            class_vertices[lab] = np.where(np.asarray(predicted) == lab)[0]
            
        return class_vertices

    def aic(self):
        
        """
        Compute Aikake Information Criterion (AIC) for each mixture model.
        """

        aic = {}
        aic = aic.fromkeys(self._labels)
        
        if not self._fitted:
            raise ValueError('Must fit model before computing AIC.')

        # for all labels in in training set
        for lab in self._labels:

            aic[lab] = self.mixtures[lab].aic(self.labelData[lab])

        return aic
            
    def bic(self):
        
        """
        Compute Bayesian Information Criterion (BIC) for each mixture model.
        """

        bic = {}
        bic = bic.fromkeys(self._labels)
        
        if not self._fitted:
            raise ValueError('Must fit model before computing AIC.')

        # for all labels in in training set
        for lab in self._labels:

            bic[lab] = self.mixtures[lab].bic(self.labelData[lab])

        return bic

##########
##########
        
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
    
    
    
##########
##########
        
class MaximumLiklihood(object):
    
    """
    Class to perform surface verex label classification based
    on maximum liklihood of the training labels.
    """
    
    def __init__(self):
        
        pass
        
    def predict(self,y,yMatch):
        
        """
        Method to predict the labels based on frequency with which test
        vertex maps to training label.
        
        Parameters:
        - - - - - 
        
            y : SubjectFeatures object for a test brain
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
        """
        
        y = ld.loadPick(y)
        yMatch = ld.loadPick(yMatch)
        
        self._predict = cu.maximumLiklihood(y,yMatch)

    def classify(self):
        '''Return the maximum liklihood label for each vertex.'''
        
        classed = {}
        classed = classed.fromkeys(self._predict.keys())
        
        for v,m in self._predict.items():
            if m:
                classed[v] = m
            else:
                classed[v] = 0
        
        self._classed = classed.values()
        
    @property
    def classed(self):
        ''' Return classified cortex.'''
        
        return self._classed
