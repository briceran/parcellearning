#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 04:20:57 2017

@author: kristianeschenburg
"""
import argparse
import sys
sys.path.append('..')

import parcellearning.matchingLibraries as lb
import glob
import os
import numpy as np
import pickle
import scipy.io as sio

subjectList = '/mnt/parcellator/parcellation/HCP/Connectome_4/SubjectList.txt'
homeDir = '/mnt/parcellator/parcellation/parcellearning/Data/'

with open(subjectList,'r') as inFile:
    s = inFile.readlines()
s = [x.strip() for x in s]

hemiFunc = {}.fromkeys(['Right','Left'])
hemiFunc['Right'] = 'R'
hemiFunc['Left'] = 'L'

hemi = 'Right'
H = hemiFunc[hemi]

"""
for train in s:
    print(train)
    
    trainLabDir = '{}Labels/HCP/'.format(homeDir)
    trainLab = '{}{}.{}.CorticalAreas.fixed.32k_fs_LR.label.gii'.format(trainLabDir,train,H)
    
    trainMidDir = '{}Midlines/'.format(homeDir)
    trainMids = '{}{}.{}.Midline_Indices.mat'.format(trainMidDir,train,H)
                
    trainSurfDir = '{}Surfaces/'.format(homeDir)
    trainSurf = '{}{}.{}.inflated.32k_fs_LR.surf.gii'.format(trainSurfDir,train,H)
    
    cond1 = True
    
    if not os.path.isfile(trainLab):
        cond1 = False
    if not os.path.isfile(trainMids):
        cond1 = False
    if not os.path.isfile(trainSurf):
        cond1 = False
    
    if cond1:   
        
        M = lb.MatchingLibraryTrain(train,trainLab,trainMids,trainSurf)
        remSubj = set(s) - set([train])
        tLab = homeDir + 'MatchingLibraries/Train/{}/'.format(hemi) + train + '.{}.MatchingLibrary.Train.p'.format(H)
        
        if not os.path.isfile(tLab) :        
            for source in remSubj:

                sourceLabDir = '{}Labels/HCP/'.format(homeDir)
                sourceLab = '{}{}.{}.CorticalAreas.fixed.32k_fs_LR.label.gii'.format(sourceLabDir,source,H)
                
                sourceMidDir = '{}Midlines/'.format(homeDir)
                sourceMids = '{}{}.{}.Midline_Indices.mat'.format(sourceMidDir,source,H)
                
                sourceSurfDir = '{}Surfaces/'.format(homeDir)
                sourceSurf = '{}{}.{}.inflated.32k_fs_LR.surf.gii'.format(sourceSurfDir,source,H)
                
                matchDir = '{}Matches/{}/'.format(homeDir,hemi)
                matchExt = '_corr*_200_50_1_50_doPCA.8.mat'
                matchString = '{}oM_{}_{}_to_{}{}'.format(matchDir,H,source,train,matchExt)
                match = glob.glob('{}'.format(matchString))

                cond2 = True
                if not os.path.isfile(sourceLab):
                    print '11'
                    print sourceLab
                    cond2 = False
                if not os.path.isfile(sourceMids):
                    print sourceMids
                    print '12'
                    cond2 = False
                if not os.path.isfile(sourceSurf):
                    print sourceSurf
                    print '13'
                    cond2 = False

                if cond2:
                    print "Building Libraries"
                    if len(match):
                        matching = match[0]
                        M.buildLibraries(source,sourceLab,sourceMids,sourceSurf,matching)

            M.saveLibraries(True,tLab)

"""

#s = ['285345']

subjectList = '/mnt/parcellator/parcellation/HCP/Connectome_4/SubjectList.txt'
homeDir = '/mnt/parcellator/parcellation/parcellearning/Data/'

with open(subjectList,'r') as inFile:
    s = inFile.readlines()
s = [x.strip() for x in s]

hemiFunc = {}.fromkeys(['Right','Left'])
hemiFunc['Right'] = 'R'
hemiFunc['Left'] = 'L'

hemi = 'Right'
H = hemiFunc[hemi]

np.random.shuffle(s)

for subj in s:
    
    print 'Current subjects: {}'.format(subj)

    # get output directory
    outLibDir = '{}MatchingLibraries/Test/'.format(homeDir)
    # output library
    outLib = '{}{}.{}.MatchingLibrary.Test.p'.format(outLibDir,subj,H)
    # output vertex library
    outVertLib = '{}{}.{}.VertexLibrary.Test.p'.format(outLibDir,subj,H)

    if not os.path.isfile(outLib):
        if not os.path.isfile(outVertLib):

            # label file
            sLabDir = '{}Labels/HCP/'.format(homeDir)
            sLab = '{}{}.{}.CorticalAreas.fixed.32k_fs_LR.label.gii'.format(sLabDir,subj,H)
    
            # midline file
            sMidDir = '{}Midlines/'.format(homeDir)
            sMid = '{}{}.{}.Midline_Indices.mat'.format(sMidDir,subj,H)
            
            # surface file
            sSurfDir = '{}Surfaces/'.format(homeDir)
            sSurf = '{}{}.{}.inflated.32k_fs_LR.surf.gii'.format(sSurfDir,subj,H)

            cond = True
            
            if not os.path.isfile(sLab):
                print 'no source label'
                cond = False
            if not os.path.isfile(sMid):
                print 'no source mids'
                cond = False
            if not os.path.isfile(sSurf):
                print 'no source surf'
                cond = False

            if cond:
    
                # Initialize test library
                N = lb.MatchingLibraryTest(subj,sLab,sMid,sSurf)
                remaining = list(set(s).difference({subj}))
    
                for train in remaining:
                    
                    print 'Current train: {}'.format(train)

                    # get 
                    trainMatchDir = '{}MatchingLibraries/Train/{}/'.format(homeDir,hemi)
                    trainMatch = '{}{}.{}.MatchingLibrary.Train.p'.format(trainMatchDir,train,H)
                    
                    matchDir = '{}Matches/{}/'.format(homeDir,hemi)
                    matchExt = '_corr*_200_50_1_50_doPCA_8.mat'
                    matches = glob.glob('{}oM_{}_{}_to_{}{}*'.format(matchDir,H,subj,train,matchExt))

                    cond2 = True
                    
                    if not os.path.isfile(trainMatch):
                        print 'no trainging object'
                        cond2 = False
                    
                    if len(matches) < 1:
                        print 'no matches'
                        cond2 = False

                    if cond2:
                        N.addToLibraries(train,trainMatch,matches[0])
    
                vertexLibrary = N.vertLib
                matchingMatrix = lb.buildMappingMatrix(vertexLibrary,180)
                print 'MatchingMatrix shape: {}'.format(matchingMatrix.shape)
                M = {}
                M['mm'] = matchingMatrix
                
                mmDir = '{}MatchingMatrices/'.format(outLibDir)
                mmOut = '{}{}.{}.MatchingMatrix.0.05.mat'.format(mmDir,subj,H)
                sio.savemat(mmOut,M)
    
                N.saveLibraries(outLib)
    
                with open(outVertLib,"wb") as output:
                    pickle.dump(N.vertLib,output,-1)
