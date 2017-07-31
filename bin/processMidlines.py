#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:47:20 2017

@author: kristianeschenburg
"""

import scipy.io as sio
import os
import numpy as np

def processMidlines(subjectList,dataDir,hemi):

    hemis = ['Left','Right']
    
    with open(subjectList,'r') as inFile:
        subjects = inFile.readlines()
    subjects = [x.strip() for x in subjects]
    
    for s in subjects:

        for h in hemis:
            if h == 'Left':
                H = 'LEFT'
                HH = 'L'
            else:
                H = 'RIGHT'
                HH = 'R'

            subjDir = '{}{}/RestingState/{}/'.format(dataDir,s,h)
            subjRest = '{}rfMRI_Z-Trans_merged_CORTEX_{}.mat'.format(subjDir,H)
            print subjRest
            outMids = '{}{}.{}.Midline_Indices.mat'.format(subjDir,s,HH);
            print outMids
            
            if os.path.isfile(subjRest):
                S = sio.loadmat(subjRest)
                rsData = S['rest']
                
                np.sum(np.abs(rsData),axis=1)
                sm = np.sum(np.abs(rsData),axis=1);
                
                mids = np.where(sm == 0)[0] + 1
                
                m = {}
                m['mids'] = mids
                
                sio.savemat(outMids,m)
        
if __name__=="__main__":
    
    dDir = '/mnt/parcellator/parcellation/HCP/Connectome_4/'
    subjList = dDir + 'SubjectList.txt'
    
    processMidlines(subjList,dDir,'R')
        