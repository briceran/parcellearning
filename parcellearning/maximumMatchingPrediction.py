#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:24:53 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.append('..')

import parcellearning.loaded as ld
import parcellearning.writeGifti as wg
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-md','--matchDir',help='Directory where matching matrices exist.',
                    required=True,type=str)
parser.add_argument('-me','--matchExt',help='Matching matrix extension.',
                    required=True,type=str)

parser.add_argument('-pd','--predDir',help='Directory to save maximum matching predictions.',
                    required=True,type=str)
parser.add_argument('-pe','--predExt',help='Maximum Matching prediction extension.',
                    required=True,type=str)

parser.add_argument('-midDir','--midlineDir',help='Directory where midlines exist.',
                    required=True,type=str)
parser.add_argument('-midExt','--midlineExt',help='Midline file extensions.',
                    required=True,type=str)

parser.add_argument('-sd','--subjectList',help='Full path to subject list file.',
                    required=True,type=str)

args = parser.parse_args()

matchDir = args.matchDir
matchExt = args.matchExt

predDir = args.predDir
predExt = args.predExt

midDir = args.midlineDir
midExt = args.midlineExt

subjectList = args.subjectList

with open(subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

hemiMap = {'Left': 'L',
           'Right': 'R'}

for subj in subjects:
    
    for h in hemiMap.keys():
        
        matchMatrix = ''.join([matchDir,subj,'.',hemiMap[h],'.',matchExt])
        midline = ''.join([midDir,subj,'.',hemiMap[h],'.',midExt])
        outFile = ''.join([predDir,subj,'.',hemiMap[h],'.',predExt])
        
        if os.path.exists(matchMatrix) and os.path.exists(midline):
            if not os.path.exists(outFile):
            
                matching = ld.loadMat(matchMatrix)
                midline = ld.loadMat(midline) - 1
                prediction = np.argmax(matching,axis=1)+1
                prediction[midline] = 0
                
                wg.writeGiftiImage(prediction,outFile,''.join(['Cortex',h]))
        
