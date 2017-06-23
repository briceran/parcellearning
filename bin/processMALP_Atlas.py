#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 00:18:52 2017

@author: kristianeschenburg
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:22:48 2017

@author: kristianeschenburg
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:26:17 2017

@author: kristianeschenburg
"""

import argparse

import sys
sys.path.append('..')

import parcellearning.malp as malp
import parcellearning.loaded as ld

import nibabel as nb
import numpy as np

import os
import pickle

parser = argparse.ArgumentParser(description='Compute random forest predictions.')
parser.add_argument('-k','--iterations', help='Range of values.',required=True)
args = parser.parse_args()
P = int(args.iterations)

dataDir = '/mnt/parcellator/parcellation/parcellearning/Data/'

subjectList = dataDir + 'HCP_Subjects.txt'

with open(subjectList,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

trainData = dataDir + 'TrainingObjects/FreeSurfer/Grouped.L.aparc.a2009s.h5'
mapData = dataDir + 'LabelAdjacency/HCP/Compiled.L.HCP.LabelAdjacency.p'

inMatchingDir = dataDir + 'MatchingLibraries/Test/'

kars = {'n_estimators': 60,
        'max_depth': 5,
        'softmax_type': 'FORESTS'}

feats = ['fs_cort','subcort','sulcal','myelin']

iters = 5
testSize = 10

inMyl = dataDir + 'MyelinDensity/285345.L.MyelinMap.32k_fs_LR.func.gii'
myl = ld.loadGii(inMyl)

Myl = nb.gifti.giftiio.read(inMyl)

for k in np.arange(P,P+1):
    
    print('Training Iteration: {}'.format(k))
    
    
    # generate list of test subjects
    testing = list(np.random.choice(subjects,size=testSize,
                                          replace=False))
    
    # generate list of train subjects
    training = list(set(subjects).difference(set(testing)))
    
    # initialize dictionary of test subject predictions
    testPredictions = {}.fromkeys(testing)
    
    kars.update({'exclude_testing': testing})

    # Option 1
    # DBSCAN = True

    kars.update({'DBSCAN': True})
    
    # initialize malp.MultiAtlas with parameters
    M = malp.Atlas(feats)
    M.set_params(**kars)
    M.initializeTraining(trainData,mapData)
    M.fit(**kars)

    print 'Training size: {}'.format(len(training))
    print 'DBSCAN status: {}'.format(kars['DBSCAN'])

    exten = '.L.MALP.atlases_{}.nEst_{}.size_{}.depth_{}.dbscan_{}'.format(1,
                                 kars['n_estimators'],
                                 len(training),
                                 kars['max_depth'],
                                 kars['DBSCAN'])
    
    outPickle = dataDir + 'CrossValidated/Iteration_{}'.format(k) + exten + '.p'
    
    for j,test_subj in enumerate(testing):

        outFunc = dataDir + 'CrossValidated/' + test_subj + exten + '.Iter_{}.func.gii'.format(k)
        
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
            
            M.predict(teobj,temps)
            Preds = M.predicted

        testPredictions[test_subj] = Preds
        Myl.darrays[0].data = np.asarray(Preds).astype(np.float32)
        nb.save(Myl,outFunc)
        
    with open(outPickle,"wb") as outFile:
        pickle.dump(testPredictions,outFile,-1)
        
        
    # Option 2
    # atlas size = 1 training subject
    # # atlases = len(training)
    # DBSCAN = False
    
    kars.update({'DBSCAN': False})
    
        # initialize malp.MultiAtlas with parameters
    M = malp.Atlas(feats)
    M.set_params(**kars)
    M.initializeTraining(trainData,mapData)
    M.fit(**kars)

    print 'Training size: {}'.format(len(training))
    print 'DBSCAN status: {}'.format(kars['DBSCAN'])

    exten = '.L.MALP.atlases_{}.nEst_{}.size_{}.depth_{}.dbscan_{}'.format(1,
                                 kars['n_estimators'],
                                 len(training),
                                 kars['max_depth'],
                                 kars['DBSCAN'])
    
    outPickle = dataDir + 'CrossValidated/Iteration_{}'.format(k) + exten + '.p'
    
    for j,test_subj in enumerate(testing):

        outFunc = dataDir + 'CrossValidated/' + test_subj + exten + '.Iter_{}.func.gii'.format(k)
        
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
            
            M.predict(teobj,temps)
            Preds = M.predicted

        testPredictions[test_subj] = Preds
        Myl.darrays[0].data = np.asarray(Preds).astype(np.float32)
        nb.save(Myl,outFunc)
        
    with open(outPickle,"wb") as outFile:
        pickle.dump(testPredictions,outFile,-1)