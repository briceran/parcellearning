#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:06:19 2017

@author: kristianeschenburg
"""


import argparse

import sys
sys.path.append('..')

import parcellearning.loaded as ld
import parcellearning.classifier_utilities as cu

import nibabel as nb
import numpy as np
import os

from sklearn import covariance

def covarianceKernel(inID,inAdj,inData):
    
    neighbors = inAdj[inID]
    covData = np.row_stack((inData[inID,:],inData[neighbors,:]))
    
    cov = covariance.empirical_covariance(covData)
    
    s = cov.shape[0]
    inds = np.triu_indices(s,1)
    
    conv = np.mean(cov[inds])
    neig = np.mean(cov[0,1:])
    
    return (conv,neig)

parser = argparse.ArgumentParser(description='Compute neighborhood variance.')
parser.add_argument('-sl','--subjectList', help='List of subjects to process.',required=True)
parser.add_argument('-inDir','--inputDirectory',help='Main input directory where data exists.',required=True)
parser.add_argument('-outDir','--outputDirectory',help='Output directory to write files to.',required=True)
parser.add_argument('-feats','--features',help='Features to compuate covariance with.',required=True)

args = parser.parse_args()

inDir = args.inputDirectory
outDir = args.outputDirectory
feats = list(args.features.split(','))

with open(args.subjectList,'r') as inFile:
    subjects = inFile.readlines();
subjects = [x.strip() for x in subjects]

inAdj = inDir + 'LabelAdjacency/L.SurfaceAdjacencyList.p'
adj = ld.loadPick(inAdj)

for s in subjects:
    
    inTrain = inDir + 'TrainingObjects/FreeSurfer/' + s + '.L.TrainingObject.aparc.a2009s.h5'
    inMyl = inDir + 'MyelinDensity/' + s + '.L.MyelinMap.32k_fs_LR.func.gii'
    
    if os.path.isfile(inTrain) and os.path.isfile(inMyl):
        
        train = ld.loadH5(inTrain,*['full']);
        train = ld.parseH5(train,feats)
        train = train[s]
        
        mergedData = cu.mergeFeatures(train,feats)
        
        myl = nb.load(inMyl)
        outVar1 = np.zeros((mergedData.shape[0],1))
        outVar2 = np.zeros((mergedData.shape[0],1))
        
        for j in np.arange(mergedData.shape[0]):
            
            [convolv, neighbs] = covarianceKernel(j,adj,mergedData)
            
            outVar1[j] = convolv
            outVar2[j] = neighbs
            
        
        myl.darrays[0].data = outVar1.astype(np.float32)
        outMyl = outDir + s + '.L.VertexVariance.Features.Convolved.func.gii'
        nb.save(myl,outMyl)
        
        myl.darrays[0].data = outVar2.astype(np.float32)
        outMyl = outDir + s + '.L.VertexVariance.Features.Neighbors.func.gii'
        nb.save(myl,outMyl)
