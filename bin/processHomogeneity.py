#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:06:49 2018

@author: kristianeschenburg
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:41:34 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.append('..')
sys.path.insert(0,'../../metrics/')
sys.path.insert(1,'../../io/')
sys.path.insert(2,'../parcellearning/')

import h5py
import pandas as pd
import numpy as np

import homogeneity as hmg
import loaded as ld
import dataUtilities as du

parser = argparse.ArgumentParser()

parser.add_argument('-fd','--featureDir',help='Path to feature data.',
                    required=True,type=str)
parser.add_argument('-fe','--featureExt',help='Feature file extension.',
                    required=True,type=str)
parser.add_argument('-pd','--parcellationDir',help='Path to parcellations.',
                    required=True,type=str)
parser.add_argument('-pe','--parcellationExt',help='Parcellation extension.',
                    required=True,type=str)

parser.add_argument('-fk','--featureKeys',help='Features to use for similarity.',
                    required=True,type=str,nargs='+')

parser.add_argument('--subjectList',help='List of subjects to process.',
                    required=True,type=str)

parser.add_argument('-out','--output',help='Output file.',required=True,type=str)

args = parser.parse_args()

featureDir = args.featureDir
featureExt = args.featureExt
featureKeys = args.featureKeys

labelDir = args.parcellationDir
labelExt = args.parcellationExt

output = args.output
f_path,f_ext = os.path.splitext(output)

with open(args.subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

cols = np.arange(1,181)

df_hmg = pd.DataFrame(columns=cols)
df_size = pd.DataFrame(columns=cols)

for subj in subjects:
    
    featureFile = ''.join([featureDir,subj,featureExt])
    labelFile = ''.join([labelDir,subj,labelExt])
    
    print featureFile
    print labelFile
    
    if os.path.exists(featureFile) and os.path.exists(labelFile):
        
        print 'Processing %s' % subj
        features = h5py.File(featureFile,mode='r')

        features = h5py.File(featureFile,mode='r')
        dataDict = features[subj]

        dataArray = du.mergeValueArrays(dataDict,keys = featureKeys)
        features.close()
        
        label = ld.loadGii(labelFile,darray=np.arange(1))
        
        regSim = hmg.regionalSimilarity(dataArray,label)
        regSize = {}.fromkeys(regSim.keys())
        
        for k in regSize.keys():
            indx = np.where(label == k)[0]
            regSize[k] = len(indx)

        df_hmg = df_hmg.append(regSim,ignore_index=True)
        df_size = df_size.append(regSize,ignore_index=True)
    else:
        print 'Files do not exist.'

print df_hmg.shape
print df_size.shape
        
df_hmg.to_csv(output)
df_size.to_csv(''.join([f_path,'.size',f_ext]))