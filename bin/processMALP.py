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

kars = {'atlas_size': 1,
        'n_estimators': 60,
        'max_depth': 5,
        'softmax_type': 'FORESTS'}

feats = ['fs_cort','subcort','sulcal','myelin']

iters = 7
testSize = 15

inMyl = dataDir + 'MyelinDensity/285345.L.MyelinMap.32k_fs_LR.func.gii'
myl = ld.loadGii(inMyl)

Myl = nb.gifti.giftiio.read(inMyl)

for k in np.arange(iters):
    
    print('Training Iteration: {}'.format(k+1))
    
    testing = list(np.random.choice(subjects,size=testSize,
                                          replace=False))
    
    testPredictions = {}.fromkeys(testing)
    
    kars.update({'exclude_testing': testing,
                 'DBSCAN': True})
    
    M = malp.MultiAtlas(feats)
    M.set_params(**kars)
    M.initializeTraining(trainData)
    
    L = len(M.datasets)
    size = M.atlas_size
    
    
    
    
    
    print 'Atlas size: {}'.format(size)
    print 'Training size: {}'.format(L)
    print 'Testing size: {}'.format(len(testing))
    print 'DBSCAN status: {}'.format(kars['DBSCAN'])
    
    Atlases = malp.parallelFitting(M,mapData,feats,**kars)

    extension = '.L.MALP.Atlases_{}.nEst_{}.Size_{}.Depth_{}.DBSCAN_{}'.format(L,
                                 kars['n_estimators'],
                                 kars['atlas_size'],
                                 kars['max_depth'],
                                 kars['DBSCAN'])
    
    outPickle = dataDir + 'CrossValidated/Iteration_{}'.format(k+1) + extension + '.p'
    
    for j,test_subj in enumerate(testing):

        outFunc = dataDir + 'CrossValidated/' + test_subj + extension + '.Iter_{}.func.gii'.format(k+1)
        
        print 'Test subject {}, {} of {}.'.format(test_subj,(j+1),len(testing))
        
        teobj = dataDir + 'TrainingObjects/FreeSurfer/' + test_subj + '.L.TrainingObject.aparc.a2009s.h5'
        temps = dataDir + 'MatchingLibraries/Test/' + test_subj + '.VertexLibrary.Test.p'
        
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
            
            Preds = malp.parallelPredicting(Atlases,teobj,temps,**kars)
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




        
    kars.update({'DBSCAN':False})

    print 'Atlas size: {}'.format(size)
    print 'Training size: {}'.format(L)
    print 'Testing size: {}'.format(len(testing))
    print 'DBSCAN status: {}'.format(kars['DBSCAN'])
    
    Atlases = malp.parallelFitting(M,mapData,feats,**kars)

    extension = '.L.MALP.Atlases_{}.nEst_{}.Size_{}.Depth_{}.DBSCAN_{}'.format(L,
                                 kars['n_estimators'],
                                 kars['atlas_size'],
                                 kars['max_depth'],
                                 kars['DBSCAN'])
    
    outPickle = dataDir + 'CrossValidated/Iteration_{}'.format(k+1) + extension + '.p'
    
    for j,test_subj in enumerate(testing):

        outFunc = dataDir + 'CrossValidated/' + test_subj + extension + '.Iter_{}.func.gii'.format(k+1)
        
        print 'Test subject {}, {} of {}.'.format(test_subj,(j+1),len(testing))
        
        teobj = dataDir + 'TrainingObjects/FreeSurfer/' + test_subj + '.L.TrainingObject.aparc.a2009s.h5'
        temps = dataDir + 'MatchingLibraries/Test/' + test_subj + '.VertexLibrary.Test.p'
        
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
            
            Preds = malp.parallelPredicting(Atlases,teobj,temps,**kars)
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
        
        
        
    
    
    kars.update({'atlas_size': L,
                 'atlases': 1,
                 'DBSCAN': False})
    
    print 'Atlas size: {}'.format(size)
    print 'Training size: {}'.format(L)
    print 'Testing size: {}'.format(len(testing))
    print 'DBSCAN status: {}'.format(kars['DBSCAN'])
    
    Atlases = malp.parallelFitting(M,mapData,feats,**kars)

    extension = '.L.MALP.Atlases_{}.nEst_{}.Size_{}.Depth_{}.DBSCAN_{}'.format(L,
                                 kars['n_estimators'],
                                 kars['atlas_size'],
                                 kars['max_depth'],
                                 kars['DBSCAN'])
    
    outPickle = dataDir + 'CrossValidated/Iteration_{}'.format(k+1) + extension + '.p'
    
    for j,test_subj in enumerate(testing):

        outFunc = dataDir + 'CrossValidated/' + test_subj + extension + '.Iter_{}.func.gii'.format(k+1)
        
        print 'Test subject {}, {} of {}.'.format(test_subj,(j+1),len(testing))
        
        teobj = dataDir + 'TrainingObjects/FreeSurfer/' + test_subj + '.L.TrainingObject.aparc.a2009s.h5'
        temps = dataDir + 'MatchingLibraries/Test/' + test_subj + '.VertexLibrary.Test.p'
        
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
            
            Preds = malp.parallelPredicting(Atlases,teobj,temps,**kars)
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
        
    
    
    
    
    kars.update({'atlas_size': L,
                 'atlases': 1,
                 'DBSCAN': True})
    
    print 'Atlas size: {}'.format(size)
    print 'Training size: {}'.format(L)
    print 'Testing size: {}'.format(len(testing))
    print 'DBSCAN status: {}'.format(kars['DBSCAN'])
    
    Atlases = malp.parallelFitting(M,mapData,feats,**kars)

    extension = '.L.MALP.Atlases_{}.nEst_{}.Size_{}.Depth_{}.DBSCAN_{}'.format(L,
                                 kars['n_estimators'],
                                 kars['atlas_size'],
                                 kars['max_depth'],
                                 kars['DBSCAN'])
    
    outPickle = dataDir + 'CrossValidated/Iteration_{}'.format(k+1) + extension + '.p'
    
    for j,test_subj in enumerate(testing):

        outFunc = dataDir + 'CrossValidated/' + test_subj + extension + '.Iter_{}.func.gii'.format(k+1)
        
        print 'Test subject {}, {} of {}.'.format(test_subj,(j+1),len(testing))
        
        teobj = dataDir + 'TrainingObjects/FreeSurfer/' + test_subj + '.L.TrainingObject.aparc.a2009s.h5'
        temps = dataDir + 'MatchingLibraries/Test/' + test_subj + '.VertexLibrary.Test.p'
        
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
            
            Preds = malp.parallelPredicting(Atlases,teobj,temps,**kars)
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
