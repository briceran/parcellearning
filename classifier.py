# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:25:55 2017

@author: kristianeschenburg
"""

from copy import deepcopy

import classifier_utilities as cu
import featureData as fd
import libraries as lb
import loaded as ld

import numpy as np
import os
import pickle

from sklearn import ensemble, covariance, neighbors, mixture, multiclass
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

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
    
    def __init__(self,trainObj,feats,scale=True):
        
        if isinstance(trainObj,str):
            trainData = ld.loadPick(trainObj)
        elif isinstance(trainObj,dict):
            trainData = trainObj
            
        if not trainData:
            raise ValueError('Training data cannot be empty.')
        
        if not feats:
            raise ValueError('Feature list cannot be empty.')
            
        if scale:
            [trainData,scalers] = fd.standardize(trainData,feats)
            
            self.scalers = scalers
            self._scaled = True            
            
        self._labels = set(cu.getLabels(trainData)) - set([0,-1])
        self.labelData = cu.partitionData(trainData,feats = feats)
        
        self.trainData = trainData
        self._features = feats

    def fit(self, model = mixture.GaussianMixture(n_components=2,covariance_type='full')):
        
        """
        Method to model the data for each training label as a mixture of
        components.  For now, we will us GaussianMixtureModels as specify the 
        number of components in each parcel.
        
        Parameters:
        - - - - - 
        
            model : dictionary of options that will be supplied to
                        sklearn.mixture.GaussianMixture()
                        
                        See sklearn documentation for more details.
                        
        Initializes:
        - - - - - -
        
            mixtures_ : dictionary mapping label to unique mixture model for 
                        data corresponding to that label
        """
        
        self._model = model

        mixtures = {}
        mixtures = mixtures.fromkeys(self._labels)

        for l in self._labels:
            
            mixtures[l] = deepcopy(model)
            mixtures[l].fit(self.labelData[l])

        self.mixtures = mixtures
        self._fitted = True
        
    def loadTest(self,y,yMatch,features,**kwargs):
        
        """
        Method to load the test data into the object.  We might be interested
        in loaded new test data into, so we have explicitly defined this is
        as a method.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain      
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
                    
            feats : names of features to include in the distance calculations
            
            kwargs : computing vertex-to-label mappings is time consuming, so
                    we allow the option to pre-load this data
        """
        
        testObject = ld.loadPick(y)

        testMatch = ld.loadPick(yMatch)
        
        if 'load' in kwargs:
            labelVertices = ld.loadPick(kwargs['load'])
            
        else:
            lMaps = testMatch.vertLib
            simplified = {n: lMaps[n].keys() for n in lMaps.keys()}
            
            labelVertices = cu.vertexMemberships(simplified,self._labels)

        self._labelVerts = labelVertices

        if self._scaled:
            toScale = {'scale': self.scalers}
            self.mergedData = cu.mergeFeatures(testObject.data,features,**toScale)
        
        self.testMatch = testMatch
        self.testObject = testObject
        
        if 'save' in kwargs:
            try:
                with open(kwargs['save'],"wb") as outFile:
                    pickle.dump(labelVertices,outFile,-1)
            except IOError:
                print('Cannot save labelVertices to file.')

    def predict(self,**kwargs):
        
        """
        Method to compute Mahalanobis distance of test data from the
        distribution of all training data for each label.
        
        Parameters:
        - - - - - 
        
        **kwargs : if power parameter is defined in kwargs, will perform
                    base classification and weighted classification of the
                    surface vertices
        """

        verts = self.testMatch.vertLib.keys()
        
        labelVerts = self._labelVerts
        mergedData = self.mergedData

        # initialize prediction dictionary
        baseline = {}
        baseline = baseline.fromkeys(verts)

        # for all labels in in training set
        for lab in self._labels:

            # compute vertices that map to that label
            members = labelVerts[lab]
                        
            if len(members) > 0:
                # compute Mahalanobis distane of vertex feature to label feaatures
                scores = self._predictPoint(mergedData,lab,members)
                
                # save results in self.predict
                baseline = cu.updateScores(baseline,lab,members,scores)
        
        self.predicted = self._classify(baseline)
        self._classified = True
                
        if kwargs:
            if kwargs['power']:
    
                weightedLL = self.weight(baseline,kwargs['power'])
                self.weighted = self._classify(weightedLL)
        
    def _predictPoint(self,data,label,members):
        
        """
        Compute GMM log-likelihood for all vertices mapped to a label.
        
        Parameters:
        - - - - - 
        
            data : merged data array
            
            label : label of interest
        
            members : vertices mapping to label
   
        """

        # get feature data of vertices
        ixData = data[members,:]
        
        logLik = self.mixtures[label].score_samples(ixData)     
        
        return logLik
    
    def weight(self,baseline,power):
        
        """
        Method to weight the score by the frequency with which
        the test vertex mapped to the training labels.
        
        Parameters:
        - - - - -
        
            base : likehoods computed without mapping frequency
            
            power : mapping frequency exponent
            
        """
        
        vertLib = self.testMatch.vertLib

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
    
    def _classify(self,scores):
        
        """
        Returns the label for which the mean Mahalanobis distance was 
        the smallest.
        
        Parameters:
        - - - - -
            
            scores : dictionary containing liklihoods of data point for each
                    label it was mapped to
        """

        predicted = {}
        predicted = predicted.fromkeys(scores.keys())

        for v,m in scores.items():
            if m:
                predicted[v] = max(m,key=m.get)
            else:
                predicted[v] = 0
                
        return predicted.values()
    
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

    def updateTrainingData(self,features):
        
        """
        Method to update the self.labelData attribute with a possibly updated
        set of feature data.
        """
        
        if set(features) != set(self.features()):
            
            # update the training data
            self.labelData = cu.partitionData(self.trainData,feats = features)
            self.features = features
            self.fit(model=self._model)
            
            # update the testing data
            self.mergedData = cu.mergeFeatures(self.testObject.data,features)
        
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

    @property
    def classes(self):
        ''' Return classified cortex.'''
        
        if not self._classified:
            raise ValueError('Run classify() first.')
        
        return self.predicted

    @property
    def features(self):
        '''Return which features are included in the model.'''
        
        return self._features
    
    @property
    def labels(self):
        '''Return training label set.'''
        
        return self._labels
    
    @property
    def labelVerts(self):
        '''Return vertex-to-label mappings.'''
        
        return self._labelVerts
        
        

##########
##########
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

##########
##########
##########
##########
        
class Train(object):
    
    """
    Class to generate trained classifiers for each label in a GroupFeatures
    object.
    
    ############
    NOT DONE YET
    ############
    
    Parameters:
    - - - - - 
    
        trainObj : GroupFeatures object containing training data and features 
                    of interest
                    
        mergedMaps : aggregated label mapping data across all training subjects
                        produced by libraries.mergeMatchingLibraryTrain
                        
        feats : name of features to include in the model
        
        scale : whether or not to scale the training data
        
    """
    
    def __init__(self,trainObj,mergedMaps,feats,scale=True,threshold = 0):
        
        if isinstance(trainObj,str):
            trainData = ld.loadPick(trainObj)
            
            if isinstance(trainData,fd.SubjectFeatures):
                trainData = cu.prepareUnitaryFeatures(trainData,feats)
                
        elif isinstance(trainObj,dict):
            trainData = trainObj
        else:
            raise ValueError('Training object is of incorrect type.')
            
        if not trainData:
            raise ValueError('Training data cannot be empty.')
        
        if not feats:
            raise ValueError('Feature list cannot be empty.')
            
        if threshold < 0 or threshold > 1:
            raise ValueError('Threshold value must be within [0,1].')
        
        if scale:
            [trainData,scalers] = fd.standardize(trainData,feats)
            
            self.scalers = scalers
            self._scaled = True
            
        self.features = feats
        self.threshold = threshold
        
        self.labels = set(cu.getLabels(trainData)) - set([0,-1])
        self.labelData = cu.partitionData(trainData,feats = feats)
        
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
            
        # rather than building the classifier training data upon loading,
        # we do not save in memory and instead build when fitting classifiers
        
        #if cond:
        #    print('Aggregating classifier training data structures.')
        #    self._buildTrainingData()        
            
    def fit(self,classifier = ensemble.RandomForestClassifier(),
            ms = 'ori',**kwargs):
        
        """
        Method to train the classifiers.
        """

        if kwargs:
            kwargs = cu.parseKwargs(kwargs)
            classifier.set_params(**kwargs)
        
        model_selector = {'oVo': OneVsOneClassifier(classifier),
                          'oVr': OneVsRestClassifier(classifier),
                          'ori': classifier}
        models = {}
        
        labelData = self.labelData
        response = self.response
        
        for l in np.arange(1,5):
            
            models[l] = deepcopy(model_selector[ms])
            
            mapped = self.mergedMappings[l]
            mapped = lb.mappingThreshold(mapped,self.threshold)
            mapped.update([l])
            
            # build classifier training data upon request
            [learned,y] = cu.mergeLabelData(labelData,response,mapped)

            models[l].fit(learned,np.squeeze(y))
        
        self.models = models
        self._fit = True
        
    def loadTest(self,y,yMatch,features,**kwargs):
        
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
                    
            feats : names of features to include in model
            
            kwargs : computing vertex-to-label mappings is time consuming, so
                    we allow the option to pre-load this data
        """
        
        testObject = ld.loadPick(y)
        testMatch = ld.loadPick(yMatch)
        
        if 'load' in kwargs:
            labelVertices = ld.loadPick(kwargs['load'])
        else:
            lMaps = testMatch.vertLib
            simplified = {n: lMaps[n].keys() for n in lMaps.keys()}
            labelVertices = cu.ertexMemberships(simplified,self._labels)

        self._labelVerts = labelVertices
        self.mergedData = cu.mergeFeatures(testObject.data,features)
        
        self.testMatch = testMatch
        self.testObject = testObject
        
        if 'save' in kwargs:
            if not os.path.isfile(kwargs['save']):
                try:
                    with open(kwargs['save'],"wb") as outFile:
                        pickle.dump(labelVertices,outFile,-1)
                except IOError:
                    print('Cannot save labelVertices to file.')

    def save(self,outFile):
        
        """
        Method to save the classification object.  This can be saved at any
        time.  However, we can only "test" the classification object if it
        has been trained.
        """
        
        if not self._fit:
            print("Warning: Classifier has not been trained yet.")
        try:
            with open(outFile,"wb") as output:
                pickle.dump(self,output,-1)
        except:
            pass
        
    def _buildTrainingData(self):
        
        """
        Method to aggregate the training data for all labels.  This will be the 
        input into self.train().
        
        The mergedMappings attribute contains the "confusion" mappings for a 
        given label.  From the surface registration step, we keep trac of
        which labels are "confused" with the target label (generally the 
        neighbors of the target label).
        """
        
        learnData = {}
        y = {}        
        
        labelData = self.labelData
        response = self.response
        
        for l in self.labels:
            # get the confusion labels for label l, add l to list
            
            mapped = self.mergedMappings[l]
            mapped = lb.mappingThreshold(mapped,self.threshold)
            
            mapped.update([l])
            
            # merge data for all labels in mapped
            [learned,resp] = cu.mergeLabelData(labelData,response,mapped)
            
            learnData[l] = learned
            y[l] = resp
            
        self.learnData = learnData
        self.y = y