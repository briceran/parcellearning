#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:26:17 2017

@author: kristianeschenburg
"""

import sys
sys.path.append('..')

import parcellearning.malp as malp
import parcellearning.loaded as ld

import nibabel as nb
import numpy as np

import os
import pickle

dataDir = '/mnt/parcellator/parcellation/parcellearning/Data/'

subjectList = dataDir + 'HCP_Subjects.txt'

with open(subjectList,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

trainData = dataDir + 'TrainingObjects/FreeSurfer/Grouped.L.aparc.a2009s.h5'
mapData = dataDir + 'LabelAdjacency/HCP/Compiled.L.HCP.LabelAdjacency.p'

inMatchingDir = dataDir + 'MatchingLibraries/Test/'

kars = {'softmax_type':'FORESTS',
        'depth':5,
        'n_estimators':60,
        'atlas_size':1}

feats = ['fs_central','subcort','sulcal','myelin']

iters = 15
testSize = 10

inMyl = dataDir + 'MyelinDensity/285345.L.MyelinMap.32k_fs_LR.func.gii'
myl = ld.loadGii(inMyl)

Myl = nb.load(myl)

for k in np.arange(iters):
    
    print('Training Iteration: {}'.format(k+1))
    
    testing = np.random.choice(subjects,size=testSize,
                                          replace=False)
    
    testPredictions = {}.fromkeys(testing)
    
    kars.update({'exclude_testing': testing})
    
    M = malp.MultiAtlas(feats)
    M.set_params(**kars)
    M.initializeTraining(trainData)
    
    L = len(M.datasets)
    size = M.atlas_size
    
    Atlases = malp.parallelFitting(M,mapData,feats)
    print('Atlas softmax type: {}.'.format(Atlases[0].softmax_type))
    
    extension = '.L.MALP.Atlases_{}.nEst_{}.Size_{}.Depth_{}'.format((k+1),
                              L,kars['n_estimators'],kars['atlas_size'],kars['depth'])
    
    outPickle = dataDir + 'CrossValidated/Iteration_{}'.format(k+1) + extension + '.p'
    
    for j,test_subj in enumerate(testing):

        outFunc = dataDir + 'CrossValidated/' + test_subj + extension + '.Iter_{}.func.gii'.format(k+1)
        
        
        print('Test subject {}, {} of {}.'.format(test_subj,j,len(testing)))
        
        teobj = dataDir + 'TrainingObjects/FreeSurfer/' + test_subj + '.L.TrainingObject.aparc.a2009s.h5'
        temps = dataDir + 'MatchingLibraries/Test/' + test_subj + '.MatchingLibrary.Test.p'
        
        cond = True
        
        if not os.path.isfile(teobj):
            cond = False
            print 'Training object for {} does not exist.'.format(test_subj)
        
        if not os.path.isfile(temps):
            cond = False
            print 'Matching library for {} does not exist.'.format(test_subj)
            
        if os.path.isfile(outFunc):
            cond = False
            print 'Label file for {} already exists.'.format(test_subj)
        
        if cond:
            
            Preds = malp.parallelPredicting(Atlases,teobj,temps)
            preds = np.column_stack(Preds)
            
            outPreds = np.zeros((myl.shape))
            
            for h in np.arange(preds.shape[0]):
            
                outPreds[h] = max(set(list(preds[h,:])),
                        key=list(preds[h,:]).count)
            
        testPredictions[test_subj] = outPreds
        
        Myl.darrays[0].data = outPreds.astype(np.float32)
        nb.save(Myl,outFunc)
        
    with open(outPickle,"wb") as outFile:
        pickle.dump(testPredictions,outFile,-1)

        
        
        
    
    

