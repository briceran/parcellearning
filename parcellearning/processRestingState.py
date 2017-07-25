#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:56:24 2017

@author: kristianeschenburg
"""

import nibabel as nb
import numpy as np
import scipy.io as sio

import os

def gifti2mat_restingState(subjectList,dataDir):
    
    with open(subjectList,'r') as inFile:
        subjects = inFile.readlines()
    subjects = [x.strip() for x in subjects]
    
    for s in subjects:
        subjDir = dataDir + s + 'RestingState/'
        subjRS = subjDir + 'rfMRI_Z-Trans_merged_CORTEX_RIGHT.gii'
        outRS = subjDir + 'rfMRU_Z-Trans_merged_CORTEX_RIGHT.mat'
        
        if os.path.isfile(subjRS):
        
            rs = nb.load(subjRS)
            
            RS = []
            
            for k in rs.darrays:
                RS.append(k.data)
            
            rData = np.column_stack(RS)
            
            rs = {}.fromkeys('rest')
            rs['rest'] = rData
            
            sio.savemat(outRS,rs);
        
if __name__=="__main__":
    
    dDir = '/mnt/parcellator/parcellation/HCP/Connectome_4/'
    subjList = dDir + 'SubjectList.txt'
    
    gifti2mat_restingState(subjList,dDir)
