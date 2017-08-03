#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 01:40:23 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')

import h5py
import numpy as np
import os

import parcellearning.loaded as ld

parser = argparse.ArgumentParser(description='Compute random forest predictions.')
# Parameters for input data
parser.add_argument('-h','--hemi',help='hemisphere to process.',type=int,required=True)
args = parser.parse_args()

hemiFunc = {}.fromkeys(['Left','Right'])
hemiFunc['Left'] = 'L'
hemiFunc['Right'] = 'R'

hemi = args.hemi
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
ptxCortExt = 'Cortical.Regionalized.ProbTrackX2.LogTransformed.Single.aparc.a2009s.mat'

ptxSubCortDir = '{}SubcorticalRegionalization/ProbTrackX2/'.format(dataDir)
ptxSubCortExt = 'SubCortical.Regionalization.ProbTrackX2.LogTransformed.Single.aparc.a2009s.mat'

midDir = '{}Midlines/'.format(dataDir)
midExt = 'Midline_Indices.mat'

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
    
    midObject = '{}{}.{}.{}'.format(midDir,s,hstr,midExt)
    
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
    if not os.path.isfile(midObject):
        print midObject
        cond = False
        
    if not cond:
        print s + ' has missing files.'
        
    if cond:
        
        mid = ld.loadMat(midObject) - 1
        
        curv = ld.loadGii(curvObject)
        if curv.ndim == 1:
            curv.shape+=(1,)
        curv[mid] = 0;
        infInd = np.isinf(curv)
        nanInd = np.isnan(curv)
        curv[infInd] = 0
        curv[nanInd] = 0
        
            
        myl = ld.loadGii(mylObject)
        if myl.ndim == 1:
            myl.shape+=(1,)
            
        myl[mid] = 0
        
        sul = ld.loadGii(sulObject)
        if sul.ndim == 1:
            sul.shape+=(1,)
        sul[mid] = 0
        infInd = np.isinf(sul)
        nanInd = np.isnan(sul)
        sul[infInd] = 0
        sul[nanInd] = 0
        
        lab = ld.loadGii(labObject)
        if lab.ndim == 1:
            lab.shape+=(1,)
        lab[mid] = 0
        infInd = np.isinf(lab)
        nanInd = np.isnan(lab)
        lab[infInd] = 0
        lab[nanInd] = 0
        
        
        fsCort = ld.loadMat(fsCortObject)
        fsCort[mid,:] = 0
        infInd = np.isinf(fsCort)
        nanInd = np.isnan(fsCort)
        fsCort[infInd] = 0
        fsCort[nanInd] = 0
        
        fsSubCort = ld.loadMat(fsSubCortObject)
        fsSubCort[mid,:] = 0
        infInd = np.isinf(fsSubCort)
        nanInd = np.isnan(fsSubCort)
        fsSubCort[infInd] = 0
        fsSubCort[nanInd] = 0
        
        ptxCort = np.log(ld.loadMat(ptxCortObject))
        ptxCort[mid,:] = 0
        infInd = np.isinf(ptxCort)
        nanInd = np.isnan(ptxCort)
        ptxCort[infInd] = 0
        ptxCort[nanInd] = 0

        ptxSubCort = np.log(ld.loadMat(ptxSubCortObject))
        ptxSubCort[mid,:] = 0;
        infInd = np.isinf(ptxSubCort)
        nanInd = np.isnan(ptxSubCort)
        ptxSubCort[infInd] = 0
        ptxSubCort[nanInd] = 0

        data = h5py.File(trainingObject,mode='w')
            
        data.create_group(s)
        data.attrs['ID'] = s
        
        data[s].create_dataset('curv',data=curv)
        print data[s]['curv'].shape
        
        data[s].create_dataset('myelin',data=myl)
        print data[s]['myelin'].shape
        
        data[s].create_dataset('sulcal',data=sul)
        print data[s]['sulcal'].shape
        
        data[s].create_dataset('label',data=lab)
        print data[s]['label'].shape
        
        data[s].create_dataset('fs_cort',data=fsCort)
        print data[s]['fs_cort'].shape
        
        data[s].create_dataset('fs_subcort',data=fsSubCort)
        print data[s]['fs_subcort'].shape
        
        data[s].create_dataset('pt_cort',data=ptxCort)
        print data[s]['pt_cort'].shape
        
        data[s].create_dataset('pt_subcort',data=ptxSubCort)
        print data[s]['pt_subcort'].shape
        
        data.close()
    
        