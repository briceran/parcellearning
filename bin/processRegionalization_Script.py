#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:21:55 2017

@author: kristianeschenburg
"""

import sys
sys.path.append('..')

import parcellearning.regionalizationMethods as RM

import h5py
import os

dataDir = '/mnt/parcellator/parcellation/parcellearning/Data/'
homeDir = '/mnt/parcellator/parcellation/HCP/Connectome_4/'

hemi = 'L'

R = 75
level = 2

subjectList = homeDir + 'SubjectList.txt'
with open(subjectList,"rb") as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

restDir = dataDir + 'RestingState/'
restExt = '.rfMRI_Z-Trans_merged_CORTEX_LEFT.mat'

midDir = dataDir + 'Midlines/'
midExt = '_Midline_Indices.mat'

layerDir = dataDir + 'BoundaryVertices/Destrieux/RegionalLayers/'
layerExt = '.' + hemi + '.aparc.a2009s.RegionalLayers.p'
#layerDir = dataDir + 'BoundaryVertices/HCP/RegionalLayers/'
#layerExt = '.L.HCP.RegionalLayers.p'

outDir = dataDir + 'CorticalRegionalization/FreeSurfer/CentralVertices/'
outExt = '.' + hemi + '.aparc.a2009s.Central.Level_2.h5'
#outDir = dataDir + 'CorticalRegionalization/HCP/CentralVertices/'
#outExt = '.L.HCP.Central.Level_2.h5'

for s in subjects:
    cond = True
    
    inLayer = ''.join([layerDir,s,layerExt])
    inRest = ''.join([restDir,s,restExt])
    inMid = ''.join([midDir,s,midExt])
    
    outFile = ''.join([outDir,s,outExt])
    
    if not os.path.isfile(inLayer):
        cond = False
        print 'Layer file for {} does not exist.'.format(s)
        
    if not os.path.isfile(inRest):
        cond = False
        print 'Resting state file for {} does not exist.'.format(s)

    if os.path.isfile(outFile):
    	cond = False
    	print 'Regional layers for {} already exist.'.format(s)
    
    if cond:
        
        print 'Computing level structures of {}.'.format(s)
        LAY = RM.regionalizeStructures(inRest,inLayer,inMid,level,R)
        
        regFile = h5py.File(outFile,'w')
        regFile['reg'] = LAY
        regFile.close()

