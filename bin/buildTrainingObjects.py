#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 01:40:23 2017

@author: kristianeschenburg
"""

import sys
sys.path.append('..')

import h5py
import numpy as np
import os

import parcellearning.loaded as ld

hemiFunc = {}.fromkeys(['Left','Right'])
hemiFunc['Left'] = 'L'
hemiFunc['Right'] = 'R'

hemi = 'Right'
hstr = hemiFunc[hemi]



baseDir = '/mnt/parcellator/parcellation/'
dataDir = '{}parcellearning/Data/'.format(baseDir)
homeDir = '{}HCP/Connectome_4/'.format(baseDir)



subjectList = '{}SubjectList.txt'.format(homeDir)
with open(subjectList,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]



troDir = '{}TrainingObjects/FreeSurfer/'.format(dataDir)
troExt = 'TrainingObject.aparc.a2009s.h5'

curDir = '{}Curvature/'.format(dataDir)
curExt = 'curvature.32k_fs_LR.shape.gii'

mylDir = '{}MyelinDensity/'.format(dataDir)
mylExt = 'MyelinMap.32k_fs_LR.func.gii'

sulDir = '{}SulcalDepth/'.format(dataDir)
sulExt = 'sulc.32k_fs_LR.shape.gii'

labDir = '{}Labels/HCP/'.format(dataDir)
labExt = 'CorticalAreas.fixed.32k_fs_LR.label.gii'

fsCortDir = '{}CorticalRegionalization/Destrieux/RestingState/'.format(dataDir)
fsCortExt = 'Cortical.Regionalized.RestingState.aparc.a2009s.mat'

fsSubCortDir = '{}SubcorticalRegionalization/RestingState/'.format(dataDir)
fsSubCortExt = 'SubCortical.Regionalization.RestingState.aparc.a2009s.mat'

ptxCortDir = '{}CorticalRegionalization/Destrieux/ProbTrackX2/'.format(dataDir)
ptxCortExt = 'Cortical.Regionalized.ProbTrackX2.aparc.a2009s.mat'

ptxSubCortDir = '{}SubcorticalRegionalization/ProbTrackX2/'.format(dataDir)
ptxSubCortExt = 'SubCortical.Regionalization.ProbTrackX2.aparc.a2009s.mat'

for s in subjects:
    
    trainingObject = '{}{}.{}.{}'.format(troDir,s,hstr,troExt)
    
    curvObject = '{}{}.{}.{}'.format(curDir,s,hstr,curExt)
    mylObject = '{}{}.{}.{}'.format(mylDir,s,hstr,mylExt)
    sulObject = '{}{}.{}.{}'.format(sulDir,s,hstr,sulExt)
    
    labObject = '{}{}.{}.{}'.format(labDir,s,hstr,labExt)
    
    fsCortObject = '{}{}.{}.{}'.format(fsCortDir,s,hstr,fsCortExt)
    fsSubCortObject = '{}{}.{}.{}'.format(fsSubCortDir,s,hstr,fsSubCortExt)
    ptxCortObject = '{}{}.{}.{}'.format(ptxCortDir,s,hstr,ptxCortExt)
    ptxSubCortObject = '{}{}.{}.{}'.format(ptxSubCortDir,s,hstr,ptxSubCortExt)
    
    cond = True

    if not os.path.isfile(curvObject):
        print curvObject
        cond = False
    if not os.path.isfile(mylObject):
        print mylObject
        cond = False
    if not os.path.isfile(sulObject):
        print sulObject
        cond = False
    if not os.path.isfile(labObject):
        print labObject
        cond = False
    if not os.path.isfile(fsCortObject):
        print fsCortObject
        cond = False
    if not os.path.isfile(fsSubCortObject):
        print fsSubCortObject
        cond = False
    if not os.path.isfile(ptxCortObject):
        print ptxCortObject
        cond = False
    if not os.path.isfile(ptxSubCortObject):
        print ptxSubCortObject
        cond = False
        
    if not cond:
        print s + ' has missing files.'
        
    if cond:
        
        curv = ld.loadGii(curvObject)
        if curv.ndim == 1:
                curv.shape+=(1,)
        myl = ld.loadGii(mylObject)
        if myl.ndim == 1:
                myl.shape+=(1,)
        sul = ld.loadGii(sulObject)
        if sul.ndim == 1:
                sul.shape+=(1,)
        lab = ld.loadGii(labObject)
        if lab.ndim == 1:
                lab.shape+=(1,)
        
        fsCort = ld.loadMat(fsCortObject)
        fsSubCort = ld.loadMat(fsSubCortObject)
        
        ptxCort = np.log(ld.loadMat(ptxCortObject))
        inds = np.isinf(ptxCort)
        ptxCort[inds] = 0
        
        ptxSubCort = np.log(ld.loadMat(ptxSubCortObject))
        inds = np.isinf(ptxSubCort)
        ptxSubCort[inds] = 0
        
        if not os.path.isfile(trainingObject):
            data = h5py.File(trainingObject,mode='w')
        else:
            data = h5py.File(trainingObject,mode='r+')
            
        data.create_group(s)
        
        data[s].create_dataset('curv',data=curv)
        data[s].create_dataset('myelin',data=myl)
        data[s].create_dataset('sulcal',data=sul)
        data[s].create_dataset('label',data=lab)
        data[s].create_dataset('fs_cort',data=fsCort)
        data[s].create_dataset('fs_subcort',data=fsSubCort)
        data[s].create_dataset('pt_cort',data=ptxCort)
        data[s].create_dataset('pt_subcort',data=ptxSubCort)
        
        data.close()
    
        