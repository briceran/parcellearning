#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:57:03 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.insert(0,'../../io/')
sys.path.insert(1,'../../metrics/')

import numpy as np
import pandas as pd

import loaded as ld
import homogeneity as hmg

parser = argparse.ArgumentParser()

parser.add_argument('-td','--trueDir',help='Path to true labels.',
                    required=True,type=str)
parser.add_argument('-te','--trueExt',help='True label extension.',
                    required=True,type=str)

parser.add_argument('-pd','--predDir',help='Path to predicted labels.',
                    required=True,type=str)
parser.add_argument('-pe','--predExt',help='Predicated label extension.',
                    required=True,type=str)

parser.add_argument('--subjectList',help='Path to list of subjects.',
                    required=True,type=str)
parser.add_argument('-out','--output',help='Name of output file.',
                    required=True,type=str)

args = parser.parse_args()

trueDir = args.trueDir
trueExt = args.trueExt

predDir = args.predDir
predExt = args.predExt

output = args.output

with open(args.subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

dice = []
for subj in subjects:
    
    trueFile = ''.join([trueDir,subj,trueExt])
    predFile = ''.join([predDir,subj,predExt])
    
    if os.path.exists(trueFile) and os.path.exists(predFile):
        true = ld.loadGii(trueFile,0)
        pred = ld.loadGii(predFile,0)
        
        inds = np.where(true > 0)[0]
        
        d = hmg.diceLabel(true[inds],pred[inds])
        dice.append(d)

DF = pd.DataFrame(columns=['Dice'])
DF['Dice'] = np.asarray(dice)
DF.to_csv(output)