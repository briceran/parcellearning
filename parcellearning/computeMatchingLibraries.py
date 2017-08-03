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

hemiFunc = {}.fromkeys(['Right','Left'])
hemiFunc['Right'] = 'R'
hemiFunc['Left'] = 'L'

hemi = 'Right'
H = hemiFunc[hemi]

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
        tLab = homeDir + 'MatchingLibraries/' + train + '.{}.MatchingLibrary.Train.p'.format(H)
        
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
                    cond2 = False
                if not os.path.isfile(sourceMids):
                    print '12'
                    cond2 = False
                if not os.path.isfile(sourceSurf):
                    print '13'
                    cond2 = False

                if cond2:
                    if len(match):
                        matching = match[0]
                        M.buildLibraries(source,sourceLab,sourceMids,sourceSurf,matching)

            M.saveLibraries(True,tLab)

for subj in s:
    
    print(subj)
    
    outLibDir = '{}MatchingLibraries/Test/'.format(homeDir)
    outLib = '{}{}.{}.MatchingLibrary.Test.p'.format(outLibDir,subj,H)
    
    outVertLib = '{}{}.{}.VertexLibrary.Test.p'.format(outLibDir,subj,H)
    
    if not os.path.isfile(outLib):

        sLabDir = '{}Labels/HCP/'.format(homeDir)
        sLab = '{}{}.{}.CorticalAreas.fixed.32k_fs_LR.label.gii'.format(sLabDir,subj,H)

        sMidDir = '{}Midlines/'.format(homeDir)
        sMid = '{}{}.{}.Midline_Indices.mat'.format(sMidDir,subj,H)
        
        sSurfDir = '{}Surfaces/'.format(homeDir)
        sSurf = '{}{}.{}.inflated.32k_fs_LR.surf.gii'.format(sSurfDir,subj,H)


        cond = True
        
        if not os.path.isfile(sLab):
            print '21'
            cond = False
        if not os.path.isfile(sMid):
            print '22'
            cond = False
        if not os.path.isfile(sSurf):
            print '23'
            cond = False
            
        print cond
            
        if cond:

            N = lb.MatchingLibraryTest(s,sLab,sMid,sSurf)

            remaining = list(set(s).difference({subj}))

            for train in remaining:
                
                trainMatchDir = '{}MatchingLibraries/Train/{}/'.format(homeDir,hemi)
                trainMatch = '{}{}.{}.MatchingLibrary.Train.p'.format(trainMatchDir,train,H)
                
                cond2 = True
                
                if not os.path.isfile(trainMatch):
                    cond2 = False
                
                matchDir = '{}Matches/{}/'.format(homeDir,hemi)
                matchExt = '_corr*_200_50_1_50_doPCA.8.mat'
                matchString = '{}oM_{}_{}_to_{}{}'.format(matchDir,H,subj,train,matchExt)
                match = glob.glob('{}'.format(matchString))
                
                if len(match) > 0 and cond2:
                    N.addToLibraries(train,trainMatch,match[0])

            vertexLibrary = N.vertLib

            N.saveLibraries(outLib)

            with open(outVertLib,"wb") as output:
                pickle.dump(N.vertLib,output,-1)
