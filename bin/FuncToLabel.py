#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 22:52:08 2017

@author: kristianeschenburg
"""

import os
import sys
sys.path.append('..')

import numpy as np

N = 10

dataDir = '/mnt/parcellator/parcellation/parcellearning/Data/'
listDir = '{}TrainTestLists/'.format(dataDir)

dataTypes = ['RestingState','Full','ProbTrackX2']
methods = ['RandomForest','NeuralNetwork','GMM']
hemis = ['L','R']

wbCommand = '/usr/bin/wb_command -metric-label-import'
inCMAP = '{}Labels/Label_Lookup_300.txt'.format(dataDir)

for itr in np.arange(N):
    
    itrDir = '{}Predictions/Model_{}/'.format(dataDir,itr)
    inList = '{}TestingSubjects.{}.txt'.format(listDir,itr)
    
    with open(inList,'r') as inSubjects:
        subjects = inSubjects.readlines()
    subjects = [x.strip() for x in subjects]
    
    for subj in subjects:
    
        for h in hemis:
            for mt in methods:
                for dt in dataTypes:
                    
                    subjFile = '{}{}.{}.{}.{}.Iteration_{}'.format(itrDir,subj,mt,h,dt,itr)
                    inFunc = '{}.func.gii'.format(subjFile)
                    outLabel = '{}.label.gii'.format(subjFile)
                    
                    if os.path.isfile(inFunc) and not os.path.isfile(outLabel):
                        runCMD = '{} {} {} {}'.format(wbCommand,inFunc,inCMAP,outLabel)
                        os.system(runCMD)
                        
                    
                    
    
    
    
    
    