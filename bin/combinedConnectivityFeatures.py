#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 00:28:28 2017

@author: kristianeschenburg
"""

import sys
sys.path.append('..')

import h5py
import numpy as np
import os
from shutil import copyfile

import parcellearning.loaded as ld

hemiFunc = {}.fromkeys(['Left','Right'])
hemiFunc['Left'] = 'L'
hemiFunc['Right'] = 'R'

baseDir = '/mnt/parcellator/parcellation/'
dataDir = '{}parcellearning/Data/'.format(baseDir)
homeDir = '{}HCP/Connectome_4/'.format(baseDir)

subjectList = '{}SubjectList.txt'.format(homeDir)

troDir = '{}TrainingObjects/FreeSurfer/'.format(dataDir)
troExt = 'TrainingObject.aparc.a2009s.h5'

fullDir = '{}FullObjects/'.format(troDir)
if not os.path.isdir(fullDir):
    os.mkdir(fullDir)

curDir = '{}/Curvature/'.format(dataDir)
curExt = 'curvature.32k_fs_LR.shape.gii'

fsSubCortDir = '{}SubcorticalRegionalization/RestingState/'.format(dataDir)
fsSubCortExt = 'SubCortical.Regionalization.RestingState.aparc.a2009s.mat'

ptxCortDir = '{}CorticalRegionalization/Destrieux/ProbTrackX2/'.format(dataDir)
ptxCortExt = 'Cortical.Regionalized.ProbTrackX2.aparc.a2009s.mat'

ptxSubCortDir = '{}SubcorticalRegionalization/ProbTrackX2/'.format(dataDir)
ptxSubCortExt = 'SubCortical.Regionalization.ProbTrackX2.aparc.a2009s.mat'

with open(subjectList,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

hemi = 'Left'
hstr = hemiFunc[hemi]

for s in subjects:
    
    trainingObject = '{}{}.{}.{}'.format(troDir,s,hstr,troExt)
    curvObject = '{}{}.{}.{}'.format(curDir,s,hstr,curExt)
    fsSubCortObject = '{}{}.{}.{}'.format(fsSubCortDir,s,hstr,fsSubCortExt)
    ptxCortObject = '{}{}.{}.{}'.format(ptxCortDir,s,hstr,ptxCortExt)
    ptxSubCortObject = '{}{}.{}.{}'.format(ptxSubCortDir,s,hstr,ptxSubCortExt)
    
    cond = True
    
    if not os.path.isfile(trainingObject):
        print trainingObject
        cond = False
    if not os.path.isfile(curvObject):
        print curvObject
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
        
    if cond:
        
        outDest = '{}{}.{}.{}'.format(fullDir,s,hstr,troExt)
        
        data = h5py.File(trainingObject,mode='r+')
        arrays = data[data.keys()[0]]
        
        del(arrays['fs_central'])
        del(arrays['vertVar'])
        
        arrays['fs_subcort'] = arrays.pop('subcort')
        
        curv = ld.loadGii(curvObject)
        fsSubCort = ld.loadMat(fsSubCortObject)
        print fsSubCort.shape
        ptxCort = ld.loadMat(ptxCortObject)
        ptxSubCort = ld.loadMat(ptxSubCortObject)
        
        arrays['fs_subcort'] = fsSubCort
        arrays['pt_cort'] = np.log(ptxCort)
        arrays['pt_subcort'] = np.log(ptxSubCort)
        arrays['curv'] = curv
        
        inds = np.isinf(arrays['pt_cort'])
        arrays['pt_cort'][inds] = 0
        
        inds = np.isinf(arrays['pt_subcort'])
        arrays['pt_subcort'][inds] = 0
        
        data.close()
        
        copyfile(trainingObject,outDest)
        