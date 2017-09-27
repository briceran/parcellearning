#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:02:25 2017

@author: kristianeschenburg
"""

from Forests import Forest

import copy
import numpy as np

from joblib import Parallel, delayed
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()


def prepareAtlasData(x_train,y_train,atlases=None,atlas_size=1):
    
    if atlas_size < 1:
        raise ValueError('Before initializing training data, atlas_size '\
                         'must be at least 1.')
    
    if atlases is not None and atlases < 0:
        raise ValueError('atlases must be positive integer or None.')

    # get list of subjecs in training data
    subjects = x_train.keys()
    
    datasets = []
    labelsets = []

    if not atlases:
        atlases = len(subjects)
    else:
        atlases = min(atlases,len(subjects))
        
    subjectSet = splitSubjects(subjects,atlas_size,atlases)
    
    for subset in subjectSet:
        td = {s: x_train[s] for s in subset}
        tl = {s: y_train[s] for s in subset}

        datasets.append(td)
        labelsets.append(tl)
        
    return [datasets,labelsets]

def splitSubjects(subjects,size,atlases):
    
    """
    Split the subject list into pieces, according to atlas size and number
    of atlases.
    
    Parameters:
    - - - - -
        subjects : list of subjects to split
        
        size : number of subjects per atlas
        
        atlases : number of atlases
    """
    
    subjectSet = []
    
    # if atlas_size == 1, there is 1 training subject per Forest
    if size == 1:
        for subj in subjects:
            subjectSet.append([subj])
    else:
        chunks = len(subjects) / size
        np.random.shuffle(subjects)
        
        if chunks >= atlases:
            for j in np.arange(atlases):
                subjectSet.append([subjects[j*size:(j+1)*size]])
        else:
            rem = atlases-chunks
            for j in np.arange(chunks):
                subjectSet.append([subjects[j*size:(j+1)*size]])
            for j in np.arange(rem):
                subset = np.random.choice(subjects,size=size,replace=False)
                subjectSet.append([subset])
    
    return subjectSet
            
        
def parallelFitting(x_train,y_train,maps,**params):

    """
    Method to fit a set of Atlas objects.
    
    Parameters:
    - - - - -
        multiAtlas : object containing independent datasets
        maps : label neighborhood map
        features : features to include in model
    """
    
    BaseAtlas = Forest()

    fittedAtlases = Parallel(n_jobs=NUM_CORES)(delayed(atlasFit)(BaseAtlas,
                             d,l,maps,**params) for d,l in zip(x_train,y_train))
    
    return fittedAtlases
    
def atlasFit(base,data,labels,maps,**kwargs):
    
    """
    Single model fitting step.
    """
    
    atl = copy.deepcopy(base)
    atl.fit(data,labels,maps,180,**kwargs)
    
    return atl


def parallelPredicting(models,x_test,match,**kwargs):
    
    """
    Method to predicted test labels in parallel
    
    Each individual model returns a vector.  These are concatenated
    column-wise and most-frequently assigned label is chosen as final label.
    """
    
    predictedLabels = Parallel(n_jobs=NUM_CORES)(delayed(atlasPredict)(models[i],
                               x_test,match,**kwargs) for i,m in enumerate(models))

    predictedLabels = np.column_stack(predictedLabels)

    classification = []
        
    for i in np.arange(predictedLabels.shape[0]):
        
        L = list(predictedLabels[i,:])
        maxProb = max(set(L),key=L.count)
        classification.append(maxProb)
    
    classification = np.asarray(classification)
    
    return predictedLabels

def atlasPredict(model,x_test,match):
    
    """
    Single model prediction step.
    """

    model.predict(x_test,match,softmax_type='FORESTS')
    
    return model.predicted

