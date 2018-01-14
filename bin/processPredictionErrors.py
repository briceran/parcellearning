#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:02:05 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.append('..')
sys.path.insert(0,'../../io/')

import parcellearning.labelAnalysis as la
import h5py
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-td','--trueDir',help='True label directory.',required=True,type=str)
parser.add_argument('-te','--trueExt',help='True label file extension.',required=True,type=str)

parser.add_argument('-pd','--predDir',help='Predicted label directory.',
                    required=True,type=str)
parser.add_argument('-pe','--predExt',help='Predicted label file extension.',
                    required=True,type=str)

parser.add_argument('-sd','--surfAdjDir',help='Surface adjacency directory.',
                    required=True,type=str)
parser.add_argument('-se','--surfAdjExt',help='Surface adjacency extension.',
                    required=True,type=str)

parser.add_argument('-md','--midDir',help='Midline directory.',required=True,type=str)
parser.add_argument('-me','--midExt',help='Midline extension.',required=True,type=str)

parser.add_argument('--subjectList',help='List of subjects to process.',
                    required=True,type=str)

parser.add_argument('-out','--output',help='Output file name.',required=True,type=str)

args = parser.parse_args()

trueDir = args.trueDir
trueExt = args.trueExt

predDir = args.predDir
predExt = args.predExt

surfAdjDir = args.surfAdjDir
surfAdjExt = args.surfAdjExt

midDir = args.midDir
midExt = args.midExt

output = args.output

with open(args.subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

for subj in subjects:
    
    inTrue = ''.join([trueDir,subj,trueExt])
    inPred = ''.join([predDir,subj,predExt])
    
    inAdj = ''.join([surfAdjDir,subj,surfAdjExt])
    inMid = ''.join([midDir,subj,midExt])
    
    errorList = []
    
    print inTrue
    print inPred
    print inAdj
    print inMid
    
    if os.path.exists(inTrue) and os.path.exists(inPred):
        if os.path.exists(inAdj) and os.path.exists(inMid):
        
            errorDistances = la.labelErrorDistances(inAdj,inTrue,inMid,inPred)
            errorList.append(errorDistances)
        
    errorList = np.asarray(np.concatenate(errorList))
    
    h5 = h5py.File(output,mode='r')
    h5.create_dataset('distances',data=errorList)
    h5.close()
    