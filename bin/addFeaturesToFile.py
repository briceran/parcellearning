#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:57:16 2017

@author: kristianeschenburg
"""

import argparse

import h5py
import sys
sys.path.append('..')

import parcellearning.loaded as ld
import os

parser = argparse.ArgumentParser(description='Add data to training object.')
parser.add_argument('-dDir','--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('-aDir','--atlasDirectory',help='Directory for specific atlas data.',required=True)
parser.add_argument('-aExt','--atlasExtension',help='Atlas-specific data extension',required=True)

parser.add_argument('-sl','--subjectList',help='List of subjects to include.',nargs='+',required=True)
parser.add_argument('-f','--featureName',help='Names of new features.',required=True)
parser.add_argument('-e','--featureExtension',help='Extension of files to be added.',required=True)
parser.add_argument('-sd','--subDirectory',help='Sub-directories where files exist.',required=True)
args = parser.parse_args()

subjectList = args.subjectList

if isinstance(subjectList,str):
    with open(subjectList,'r') as inFile:
        subjects = inFile.readlines()
    subjects = [x.strip() for x in subjects]
    
elif isinstance(subjectList,list):
    subjects = subjectList
    
print 'len subjects: ' + str(len(subjects))

dataDir = args.dataDirectory
atlasDir = args.atlasDirectory
trainingDirectory = dataDir + atlasDir
atlasExt = args.atlasExtension

subDir = args.subDirectory
feature = args.featureName
fExt = args.featureExtension

funcs = {'gii': ld.loadGii,
         'mat': ld.loadMat,
         'nii': ld.loadMat
         }

for s in subjects:
    
    print s
    
    inTrain = trainingDirectory + s + atlasExt
    inFeatr = dataDir + subDir + s + fExt
    
    if os.path.isfile(inTrain) and os.path.isfile(inFeatr):
        
        train = h5py.File(inTrain,mode='r+')
        fileParts = str.split(inFeatr,'.')
        
        featureData = funcs[fileParts[-1]](inFeatr)
        train[s].create_dataset(feature,data=featureData)
        
        train.close()   