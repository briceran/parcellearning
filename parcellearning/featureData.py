# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:36:05 2017

@author: kristianeschenburg
"""

from copy import deepcopy

import loaded as ld

from sklearn import preprocessing

import h5py
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
        self.featNames = []
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
            print('Warning: Data attributes have unequal number of obsevations.  '
                  'Check input data.')
            
            
    def addFeature(self,featName,featPath):
        
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

        tempData = loadFeature(featPath)
        
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
                
            self.data.update({featName: tempData})
            self.featNames.append(featName)
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
                print('Feature {} has {} observations -- expecting '
                      '{}.'.format(k,v.shape[0],self.obs))
                cond = False
        
        return cond
    

class GroupFeatures(object):
    
    """
    Class to compile feature data from a set of subjects.
    
    Class expects that file structure for a set of subjects to be the same.  
    The individual subject feature matrices have been generated, 
    and saved somewhere.  We will save them as HDF5 files.
    
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
        
        if (str.split(exten,'.')[-1] != "h5"):
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
            
        trainData = {}
        testData = {}
                    
        # loop over subject IDs
        for subject in self.subjects:
                        
            inFile = ''.join([self.inputDir,subject,self.exten])
            sFeats = ld.loadH5(inFile,*['full'])
            
            ID = sFeats.attrs['ID']
            train = sFeats.attrs['train']
                
            # check to make sure subject ID is same as input subject
            if ID != subject:
                raise Exception('Data file ID {} does not match the input '
                                'subject name {}.'.format(sFeats.ID,subject))
                
            # if subject data has correct number, names, and sizes of 
            # features
            if self.checkFeatureNames(sFeats) and self.checkFeatureSizes(sFeats):
                # split data into training and testing types
                if train:
                    trainData[subject] = {}.fromkeys(self.features)
                    
                    for f in self.features:
                        trainData[subject][f] = sFeats[f]
                        
                else:
                    testData[subject] = {}.fromkeys(self.features)
                    
                    for f in self.features:
                        testData[subject][f] = sFeats[f]
        
        self.training = trainData
        self.testing = testData
                        

    def checkFeatureNames(self,data):
        
        """
        Method to compare the input data to what is expected by the user.
        
        self.feature of "GroupFeatureData" object should match the
        self.featureNames of each "SubjectFeatureData" object read in.
        
        Feature counts and feature names should match.
        """
        
        cond = True
        train = data.attrs['train']
        ID = data.attrs['ID']
        
        dataFeatures = set(data.keys()).difference({'label'})
        groupFeatures = self.features.keys()
        
        # determine if subject is training or testing data
        if train:
            kind = "Train"
        else:
            kind = "Test"

        # make sure the features names are as expected
        if set(groupFeatures) != set(dataFeatures):
            print('Warning: {} subject {} does not have the same feature'
                  'names as specified by user.  It will not be added to '
                  'the compiled data'.format(kind,ID))
            cond = False
        
        return cond

    def checkFeatureSizes(self,data):
        
        """
        Method to check the dimensionality of features in the aggregated 
        training and testing data.
        """
        
        cond = True
        ID = data.attrs['ID']
        
        dataFeatures = set(data.keys()).difference({'label'})
        features = self.features
                
        # loop through the expected features
        for f in features:

            # check if current feature exists in subject features
            if f in dataFeatures:
                            
                if data[f].ndim == 1:
                    data[f].shape += (1,)
                    
                yDim = data[f].shape[1]

                # check that y-dim is same as expected number of regions
                if yDim != features[f]:
                    print('Subject {}, feature {}, has size of {}, '
                          'but expecting {}.'.format(ID,f,yDim,features[f]))
                    cond = False
            else:
                print('Subject {} does not have feature {}.'.format(ID,f))
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
                    print('Cannot save to file.')
            else:
                print('There is no training data to write.')
        
        if outputs.has_key('test'):
            if self.testing:
                try:
                    with open(outputs['test'],"wb") as testOut:
                        pickle.dump(self.testing,testOut,-1)
                except IOError:
                    print('Cannot save to file.')
            else:
                print('There is no testing data to write.')
                

def addToFeaturesObject(featureObject,newFeats):
    
    """
    Method to add to a SubjectFeature object that has already been loaded
    and saved.
    
    Parameters:
    - - - - - 
    
        featureObject : (str,SubjectFeature object)
        
        newFeats : dictionary of feature name and path to feature
        
    """

    if isinstance(featureObject,str):
        k = ['full']
        obj = ld.loadH5(featureObject,*k)
    elif isinstance(featureObject,h5py._hl.files.File):
        obj = featureObject
    else:
        raise TypeError('featureObject cannot be read.')
        
    obs = obj.attrs['obs']
    featNames = list(obj.attrs['featNames'])
        
    # loop over all key-value pairs in new feats
    for fn,fp in newFeats.items():
        
        print('feat name: ',fn)

        # make sure feature doesn't already exist
        # dont add if it does
        if fn not in obj and fn not in obj.attrs:
            
            # load data
            tempData = loadFeature(fp)
            print('data shape: ',tempData.shape)
            
            if checkFeatureSize(obs,tempData) and checkFeatureType(tempData):
                print('conditions true')

                # update feature list and number of eatures
                obj.attrs['numFeatures'] += 1
                
                featNames.append(fn)
                obj.attrs['featNames'] = featNames
                
                obj[fn] = tempData
        else:
            print('{} feature already exists.'.format(fn))
    
    return obj
    
def loadFeature(path):
    
    """
    Method to get data from a single feature.
    
    Parameters:
    - - - - - 

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
    tempData = functions[parts[-1]](path)

    return tempData

def saveSubjectFeatures(featureObject,outFile):
        
    """
    Method to dump features into a pickle file.  Generally, we expect that
    the HDF5 files will be saved as "${SUBJECT_ID}.${extension}.h5" such
    that they can be easily loaded later on.
    """
    
    obj = featureObject
    
    if len(obj.data.keys()) == 0 or not hasattr(obj,'data'):
        print('Data attribute for {} is empty.  Cannot save.'.format(obj.ID))
    else:
        if obj.compareFeatureSamples():

            outFeatures = h5py.File(outFile,mode='w')
            
            for k in obj.data.keys():
                outFeatures.create_dataset(k, data=obj.data[k]);
                
            atrribs = set(obj.__dict__).difference({'data','features'})
            
            for attr in atrribs:
                outFeatures.attrs[attr] = obj.__dict__[attr];
            
            outFeatures.close()
            
def saveGroupFeatures(featureObject,outputKeys):
    
    """
    Method to dump training and testing GroupFeatures data into their own HDF5 objects.
    """
    
    obj = featureObject
    features = obj.features
    
    if 'train' in outputKeys:
        
        training = obj.training
        outFile = h5py.File(outputKeys['train'],mode='w')
        
        trainSubjects = training.keys()
            
        for s in trainSubjects:
            outFile.create_group(s)
            
            for f in features:
                outFile[s].create_dataset(f,data=training[s][f])
            
        outFile.attrs['numSubjects'] = len(trainSubjects)
        outFile.close()
    
    if 'test' in outputKeys:
        
        testing = obj.testing
        outFile = h5py.File(outputKeys['test'],mode='w')

        testSubjects = testing.keys()

        for s in testSubjects:
            outFile.create_group(s)

            for f in features:
                outFile[s].create_dataset(f,data=testing[s][f])

        outFile.attrs['numSubjects'] = len(testSubjects)
        outFile.close()
        

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


def standardize(grouped,features):

    """
    Method to demean the data from a GroupFeatures object.  This object is
    just a dictionary of dictionaries -- each main key is a subject ID, with
    sub-keys correpsonding to features i.e. resting-state, cortical metrics.
    
    Standardization is performed upon run-time -- we might want to save the
    mean and variance of each feature, and will return these, along with a 
    demeaned and unit-varianced GroupFeatures object.
    
    Parameters:
    - - - - -
        grouped : pickle file -- output of GroupFeatures.save(
                    {'train': 'outputFile.p'})
    """
    
    if isinstance(grouped,str):
        trainData = ld.loadPick(grouped)
    elif isinstance(grouped,dict):
        trainData = deepcopy(grouped)
    else:
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
        