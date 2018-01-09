#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:40:24 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.insert(0,'../../io/')

import loaded as ld

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-labDir','--labelDir',help='Directory where predicted labels exist.',
                    required=True,type=str)
parser.add_argument('-labExt','--labelExt',help='Predicted label extension.',
                    required=True,type=str)
parser.add_argument('-truDir','--trueDir',help='Directory where true labels exist.',
                     required=True,type=str)
parser.add_argument('-truExt','--trueExt',help='True label extension.',
                    required=True,type=str)
parser.add_argument('--subjectList',help='List of subjects.',required=True,type=str)
parser.add_argument('-out','--output',help='Output file name.',
                    required=True,type=str)

args = parser.parse_args()

labDir = args.labelDir
labExt = args.labelExt

trueDir = args.trueDir
trueExt = args.trueExt

output = args.output

df = pd.DataFrame(columns=['accuracy'])

with open(args.subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

for subj in subjects:
    
    predLabel = ''.join([labDir,subj,labExt])
    trueLabel = ''.join([trueDir,subj,trueExt])
    
    if os.path.exists(predLabel) and os.path.exists(trueLabel):
        
        pred = ld.loadGii(predLabel,darray=np.arange(1))
        true = ld.loadGii(trueLabel,darray=np.arange(1))
        
        trueInds = np.where(true > 0)[0]
        
        accuracy = np.mean(pred[trueInds] == true[trueInds])
        df = df.append({'accuracy': accuracy},ignore_index=True)

df.to_csv(output)
        
        
