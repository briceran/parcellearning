# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:36:05 2017

@author: kristianeschenburg
"""

from copy import deepcopy

import loaded as ld

from sklearn import preprocessing

import numpy as np
import os
import pickle

class SubjectFeatures(object):
    
    """
    Class to build feature data for a single subject.  We represent this data
    as a dictionary.  This will allow us to "match" features
    across subjects later on by comparing their keys and values.
    
    Parameters:
    - - - - - 
        subject : subject ID
        
        features : dictionary, where keys are "names" of features (ex.
                    "reg" for regionalized connectivity, "spect" for 
                    spectral features, etc.), and values are paths to the 
                    data
                    
                    ex. features = {"reg" : /path/to/reg_matrix.mat, 
                    "spect" : /path/to/spectral_features.mat}
                    
        obs : expected number of observations per feature type
        
        train : boolean indicating this is a training subject -- default
                is False.  If set to True, must also provide a label file 
                with the key in features as "label".
    """
    
    def __init__(self,subject,features,obs,train = False):
        
        # if train == True, must have a label file
        if train and "label" not in features:
            raise Exception("If train == True, label file must be provided.")
            train = False
        
        self.ID = subject
        self.features = features
        self.obs = obs
        
        self.train = train
        self.numFeatures = 0
        
    def loadFeatures(self):
        
        """
        Method to build a dictionary of data containing each of the keys
        included in self.features.
        
        The dictionary is represented as a set of keys as defined originally
        in self.features.  Ex
            
        data = {'reg' : N x 75 matrix, 'spect' : N x K matrix...}
            
        We will add a check here at the subject level to make sure each 
        feature has the same number of observations.  If they don't, a flag 
        will be raised.
        """

        # loop over all features
        for fn,fp in self.features.items():
            
            self.addFeature(fn,fp)

        # check if all features have same number of samples
        if not self.compareFeatureSamples():
            print('Warning: Data attributes have unequal number of ' \
                            'obsevations.  Check input data.')
            
            
    def addFeature(self,featName,featPath,*args):
        
        """
        Method to add a new feature to a loaded feature object.  If loading a HDF5 file,
        expects that there is only 1 attribute in the file (num(featPath.keys()) = 1).

        While loaded.loadH5 can return a dictionary of attributes, addFeature expects a
        single feature.
        
        Parameters:
        - - - - -
            featName : name of feature to include (i.e. 'reg', 'label')
                        SubjectFeatures object
            
            featPath : path where feature exists


        """

        tempData = getSingleFeature(featName,featPath,*args)
        
        cond = True
        if not checkFeatureType(tempData):
            print('Feature {} is not a numpy.ndarray.\n'.format(featName))
            cond = False
        else:
            if tempData.ndim == 1:
                tempData.shape += (1,)
        
        if not checkFeatureSize(self.obs,tempData):
            print('Feature {} has incorrect observation count.\n'.format(featName))
            cond = False
        
        if cond:
            if not hasattr(self,'data'):
                self.data = {}
            
            self.data[featName] = tempData
            self.numFeatures += 1
            
            
    def removeFeature(self,featName):
        
        """
        Method to remove a feature from a loaded feature object.
        """
        
        if featName in self.data.keys():
            del self.data[featName]
            self.numFeatures += -1
        else:
            print('Feature {} does not exist.')
                
            
    def compareFeatureSamples(self):
        
        """
        Method to check consistency in the loaded feature observation counts.
        Returns True if all features have same number of observations.
        
        All data must be in numpy.array format, in order to have the correct
        size format.
        """
        
        # get set of possible number of samples
        # expect that there should only be number of samples
        
        cond = True

        # loop over each feature, get number of samples, add to set
        for (k,v) in self.data.items():
            
            if v.shape[0] != self.obs:
                print('Feature {} has {} observations -- expecting '\
                                '{}.'.format(k,v.shape[0],self.obs))
                cond = False
        
        return cond
    
    def saveFeatures(self,outFile):
        
        """
        Method to dump features into a pickle file.  Generally, we expect that
        the Pickle files will be saved as "${SUBJECT_ID}.${extension}.p" such
        that they can be easily loaded later on.
        """
        
        # check if class has attribute data
        if len(self.data.keys()) == 0 or not hasattr(self,'data'):
            print('Data attribute is empty. Add data to object '\
                  'before saving.'.format(self.ID))
        else:
            # check if all features have the same number of samples
            if self.compareFeatureSamples():
                try:
                    with open(outFile,"wb") as output:
                        pickle.dump(self,output,-1)
                except IOError:
                    print('Cannot save object to filename.')
            else:
                print('Warning: self.data has unequal numbers of observations.  ' \
                                'Cannot save.  Check the feature sizes.')
    

class GroupFeatures(object):
    
    """
    Class to compile feature data from a set of subjects.
    
    Class expects that file structure for a set of subjects to be the same.  
    The individual subject feature matrices have been generated, 
    and saved somewhere.  For now, we will save them as Pickle files.
    
    Parameters:
    - - - - -
        features : dictionary of features expected in each subject, and
                    values corresponding to each feature's dimensionality
    """
    
    def __init__(self,features):
        
        if not features:
            print('Warning: Features cannot be empty.')
        else:
            self.features = features
            
        self.count = len(self.features.keys())
        
        self.training = {}
        self.testing = {}
        
    def compileFeatures(self,subjects,inputDir,exten):
        
        """
        
        Method to seperate subjects based on training / testing
        characteristics.
        
        Parameters:
        - - - - -
            subjects : array of subject IDs to be compiled.  Class distinguishes
                        between BuildSubjectFeatureData.train == True // False.
                        Subjects with same value of boolean "train" 
                        will be  compiled together.
                        
            inputDir : directory where the subject feature matrices are saved.  
                        If generating multiple types of feature matrices 
                        (i.e. different features), make a new directory.
            
            exten : extension of BuildSubjectFeatureData pickle files
        """
        
        if (str.split(exten,'.')[-1] != "p"):
            print('Warning: Incorrect file extension.')
        else:
            self.exten = exten
                        
        if not subjects:
            print('Warning: Input subject cannot be empty.')
        else:
            self.subjects = subjects
                        
        if not os.path.isdir(inputDir):
            print('Warning: Input directory does not exist.')
        else:
            self.inputDir = inputDir
                    
        # loop over subject IDs
        for subject in self.subjects:
                        
            inFile = self.inputDir + subject + self.exten
            
            feats = ld.loadPick(inFile)
                
            # check to make sure subject ID is same as input subject
            if feats.ID != subject:
                raise Exception('Data file ID {} does not match the input '\
                                'subject name {}.'.format(feats.ID,subject))
                
            # if subject data has correct number, names, and sizes of 
            # features
            if self.checkFeatureNames(feats) and self.checkFeatureSizes(feats):
                # split data into training and testing types
                if feats.train:
                    self.training[subject] = feats.data
                else:
                    self.testing[subject] = feats.data

    def checkFeatureNames(self,data):
        
        """
        Method to compare the input data to what is expected by the user.
        
        self.feature of "GroupFeatureData" object should match the
        self.featureNames of each "SubjectFeatureData" object read in.
        
        Feature counts and feature names should match.
        """
        
        cond = True
        dataFeatures = set(data.data.keys()) - set(["label"])
        
        # determine if subject is training or testing data
        if data.train:
            kind = "Train"
            enum = data.numFeatures - 1
        else:
            kind = "Test"
            enum = data.numFeatures
        
        # make each subject has the correct number of features
        if enum != self.count:
            print('Warning: {} subject {} does not '\
                            'have the same number of features as '\
                            'specified by user.  It will not be added to '\
                            'the compiled data.'.format(kind,data.ID))
            cond = False
                    
        # make sure the features names are as expected
        if set(self.features.keys()) != set(dataFeatures):
            print('Warning: {} subject {} does not '\
                            'have the same feature names as specified by user. '\
                            'It will not be added to the compiled '\
                            'data'.format(kind,data.ID))
            cond = False
        
        return cond

    def checkFeatureSizes(self,data):
        
        """
        Method to check the dimensionality of features in the aggregated 
        training and testing data.
        """
        
        cond = True
        dataFeatures = set(data.data.keys()) - set(["label"])
                
        # loop through the expected features
        for f in self.features.keys():

            # check if current feature exists in subject features
            if f in dataFeatures:
                            
                if data.data[f].ndim == 1:
                    data.data[f].shape += (1,)
                
                size = data.data[f].shape[1]
                
                # check that y-dim is same as expected number of regions
                if size != self.features[f]:
                    print('Subject {}, feature {} has size of {} -- '\
                                    'expecting {}.'.format(data.ID,f,size,self.features[f]))
                    cond = False
            else:
                print('Subject {} does not have feature {}.'.format(data.ID,f))
                cond = False

        return cond

    def saveData(self,outputs):
        
        """
        Method to save groupwise training or testing data.
        
        Parameters:
        - - - - - 
            outputs : dictionary, where keys are 'train' or 'test', and the
                        values are output file names.
        """
                
        # If user wants to write training data
        # check that correct key is specified
        if outputs.has_key('train'):
            if self.training:
                try:
                    with open(outputs['train'],"wb") as trainOut:
                        pickle.dump(self.training,trainOut,-1)
                except IOError:
                    print('Cannot save to file {}.'.format(trainOut))
            else:
                print('There is no training data to write.')
        
        if outputs.has_key('test'):
            if self.testing:
                try:
                    with open(outputs['test'],"wb") as testOut:
                        pickle.dump(self.testing,testOut,-1)
                except IOError:
                    print('Cannot save to file {}.'.format(testOut))
            else:
                print('There is no testing data to write.')
                
                
def addToFeatureObject(featureObject,newFeats):
    
    """
    Method to add to a SubjectFeature object that has already been loaded
    and saved.
    
    Parameters:
    - - - - - 
    
        featureObject : SubjectFeature object
        
        newFeats : dictionary of feature name and path to feature
        
    """
    
    obj = ld.loadPick(featureObject)
    
    for fn,fp in newFeats.items():
        
        obj.addFeature(fn,fp)
    
    obj.saveFeatures(featureObject)
    
    
def checkFeatureSize(N,data):
    
    """
    Boolean to check whether feature has correct number of observations.
    """
    
    return N == data.shape[0]

def checkFeatureType(data):
    
    """
    Boolean to check whether feature is of type numpy.array
    """
    
    return isinstance(data,np.ndarray)


def getSingleFeature(feat,path,*args):
    
    """
    Method to get data from a single feature.
    
    Parameters:
    - - - - - Gr
    
        feat : feature name
        
        path : path to feature data

        *args : (list,str) if loading a HDF5 file, or (int) if loading Gifti file

    """
    
    # dictionary of functions, call depends on file extension
    functions = {'gii' : ld.loadGii,
                 'h5' : ld.loadH5,
                 'mat' : ld.loadMat,
                 'nii' : ld.loadGii,
                 'p' : ld.loadPick}
    
    # get file extension
    parts = str.split(path,'.')
    
    # temporarilly load data
    tempData = functions[parts[-1]](path,*args)

    return tempData

def standardize(grouped,features):
    
    """
    Method to standardize the data from a GroupFeatures object.  This object is
    just a dictionary of dictionaries -- each main key is a subject ID, with
    sub-keys correpsonding to features i.e. resting-state, cortical metrics.
    
    Standardization is performed upon run-time -- we might want to save the
    mean and variance of each feature, and will return these, along with a 
    standardized GroupFeatures object.
    
    Parameters:
    - - - - -
        grouped : pickle file -- output of GroupFeatures.save(
                    {'train': 'outputFile.p'})
    """
    
    if isinstance(grouped,str):
        trainData = ld.loadPick(grouped)
    elif isinstance(grouped,dict):
        trainData = deepcopy(grouped)
    
    if not trainData:
        raise ValueError('Training data cannot be empty.')
        
    subjects = trainData.keys()
    
    mappings = {}
    mappings = mappings.fromkeys(subjects)
    
    scalers = {}
    scalers = scalers.fromkeys(features)
    
    scale = preprocessing.StandardScaler(with_mean=True,with_std=True)
    
    for f in features:

        c = 0
        tempData = []
        scalers[f] = deepcopy(scale)
        
        for s in subjects:
            
            mappings[s] = {}
            
            subjData = trainData[s][f]
            [x,y] = subjData.shape
            mappings[s]['b'] = c
            mappings[s]['e'] = c+x
            
            tempData.append(subjData)
            c += x
        
        tempData = np.row_stack(tempData)
        tempData = scalers[f].fit_transform(tempData)
        
        for s in subjects:
            
            coords = mappings[s]
            trainData[s][f] = tempData[coords['b']:coords['e'],:]
    
    return(trainData,scalers)
        