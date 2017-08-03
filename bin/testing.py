#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:58:44 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')


import parcellearning.matchingLibraries as lb
import glob
import os

subjectList = '/mnt/parcellator/parcellation/HCP/Connectome_4/SubjectList.txt'
homeDir = '/mnt/parcellator/parcellation/parcellearning/Data/'

with open(subjectList,'r') as inFile:
    s = inFile.readlines()
s = [x.strip() for x in s]


S = '285345'

remain = list(set(s).difference({S}))

outLibDir = '{}MatchingLibraries/Test/'.format(homeDir)
outLib = '{}{}.{}.MatchingLibrary.Test.p'.format(outLibDir,S,'R')
outVertLib = '{}{}.{}.VertexLibrary.Test.p'.format(outLibDir,S,'R')

# label file
sLabDir = '{}Labels/HCP/'.format(homeDir)
sLab = '{}{}.{}.CorticalAreas.fixed.32k_fs_LR.label.gii'.format(sLabDir,S,'R')

# midline file
sMidDir = '{}Midlines/'.format(homeDir)
sMid = '{}{}.{}.Midline_Indices.mat'.format(sMidDir,S,'R')

# surface file
sSurfDir = '{}Surfaces/'.format(homeDir)
sSurf = '{}{}.{}.inflated.32k_fs_LR.surf.gii'.format(sSurfDir,S,'R')

if not os.path.isfile(outLib):

    N = lb.MatchingLibraryTest(S,sLab,sMid,sSurf)
    
    for tr in remain:
        
        trainMatchDir = '{}MatchingLibraries/Train/{}/'.format(homeDir,'Right')
        trainMatch = '{}{}.{}.MatchingLibrary.Train.p'.format(trainMatchDir,tr,'R')
        
    
        if os.path.isfile(trainMatch):
            print trainMatch
            matchDir = '/mnt/parcellator/parcellation/parcellearning/Data/Matches/Right/'
            matchExt = '_corr*_200_50_1_50_doPCA_8.mat'
            matchString = '{}oM_{}_{}_to_{}{}'.format(matchDir,'R',S,tr,matchExt)
            
            print matchString
            match = glob.glob(matchString)
            print len(match)
            
            if len(match) > 0:
                N.addToLibraries(tr,trainMatch,match[0])