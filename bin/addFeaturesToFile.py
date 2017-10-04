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

"""
dataDirectory : main directory where data exists
subjectList : list of subjects to apply add/remove feature to
toAddName : names of features to add to training object
toAddExtension : file extensions of data to add
toAddDir : sub-directories where new data exists within dataDirectory
toRemoveName : names of features to remove from training object
"""

parser = argparse.ArgumentParser(description='Add data to training object.')
parser.add_argument('--dataDirectory',help='Directory where data exists.',required=True)

parser.add_argument('--hemisphere',help='Hemisphere to process.',type=str,required=True)

parser.add_argument('--subjectList',help='List of subjects to include.',nargs='+',required=True)

parser.add_argument('--toAddName',help='Names of new features.',type=str,required=True)
parser.add_argument('--toAddExtension',help='Extension of files to be added.',type=str,required=True)
parser.add_argument('--toAddDir',help='Sub-directories where files exist.',type=str,required=True)

parser.add_argument('--toRemoveName',help='Names of features to remove from training object.',required=False)

args = parser.parse_args()

subjectList = args.subjectList

cond = True
try:
    parts = str.split(subjectList[0],'.')
    inputList = subjectList[0]
except:
    print 'Failed'
    cond = False

if cond:
    with open(inputList,'r') as inFile:
        subjects = inFile.readlines()
    subjects = [x.strip() for x in subjects]
else:
    subjects = subjectList

print 'len subjects: ' + str(len(subjects))

dataDir = args.dataDirectory
trainingDirectory = ''.join([dataDir,'TrainingObjects/'])
trainingExtension = ''.join(['.',args.hemisphere,'.','TrainingObject.h5'])

featureDirectory = ''.join([args.toAddDir,'/'])
featureExtension = ''.join(['.',args.hemisphere,'.',args.toAddExtension])
feature = args.toAddName

funcs = {'gii': ld.loadGii,
         'mat': ld.loadMat,
         'nii': ld.loadMat
         }

for s in subjects:
    
    print 'Adding {} to subject {}'.format(feature,s)
    inObject = ''.join([trainingDirectory,s,trainingExtension])
    inFeature = ''.join([trainingDirectory,featureDirectory,s,featureExtension])

    if os.path.isfile(inObject) and os.path.isfile(inFeature):
        
        train = h5py.File(inObject,mode='r+')
        fileParts = str.split(inFeature,'.')
        
        featureData = funcs[fileParts[-1]](inFeature)
        train[s].create_dataset(feature,data=featureData)
        
        train.close()