#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:17:23 2017

@author: kristianeschenburg
"""

import loaded as ld
import nibabel as nb
import numpy as np
import os

def processLabelResizing(subjectList,dataDir,hemi):
    
    """
    
    """

    with open(subjectList,'r') as inFile:
        subjects = inFile.readlines()
    subjects = [x.strip() for x in subjects]
    
    hcpDir = dataDir + 'Haynor/Connectome_4_With_MMP/Labels/'
    midDir = dataDir + 'parcellearning/Data/Midlines/'
    funDir = dataDir + 'HCP/Connectome_4/'
    outDir = dataDir + 'parcellearning/Labels/HCP/'
    
    lExt = '.' + hemi + '.CorticalAreas_dil_NoTask_Final_Individual.32k_fs_LR.dlabel.nii'
    fExt = '.' + hemi + '.MyelinMap.32k_fs_LR.func.gii'
    mExt = '.' + hemi + '.Midline_Indices.mat'
    
    N = 32492
    
    inCMAP = hcpDir + 'Label_Lookup_300.txt'
    
    for s in subjects:
        
        inLabel = hcpDir + s + lExt
        inMid = midDir + s + mExt
        inFunc = funDir + s + '/Surface/MNI/' + s + fExt
        
        print inLabel
        print inMid
        print inFunc
        
        if os.path.isfile(inLabel) and os.path.isfile(inMid) and os.path.isfile(inFunc):
            
            print s
            
            outFunc = outDir + s + '.' + hemi + '.CorticalAreas.fixed.32k_fs_LR.func.gii'
            outLabel = outDir + s + '.' + hemi + '.CorticalAreas.fixed.32k_fs_LR.label.gii'
            
            label = ld.loadGii(inLabel)
            mids = ld.loadMat(inMid) - 1
            
            fixed = ld.fixLabelSize(mids,label,N)
            
            M = nb.load(inFunc)
            M.darrays[0].data = np.asarray(fixed.astype(np.float32))
            
            nb.save(M,outFunc)
            
            cmd = '/usr/bin/wb_command {} {} {}'.format(outFunc,inCMAP,outLabel)
            
            os.system(cmd)
        
        

if __name__=="__main__":
    
    
    dDir = '/mnt/parcellator/parcellation/'
    subjList = dDir + 'HCP/Connectome_4/SubjectList.txt'

    processLabelResizing(subjList,dDir,'R')