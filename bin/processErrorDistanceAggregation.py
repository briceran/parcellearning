#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:01:30 2018

@author: kristianeschenburg
"""


import h5py
import numpy as np

power = [0,1]
hemi = ['L','R']
models = ['GMM','NeuralNetwork','RandomForest']
data = ['Full','RestingState','ProbTrackX2']
x = np.arange(5)

inDir = '/mnt/parcellator/parcellation/parcellearning/Data/Predictions/'

for p in power:
    for h in hemi:
        for m in models:
            for dT in data:
                df = []

                for iters in x:
                    
                    mDir = ''.join([inDir,'Model_{}/ErrorMaps/'.format(iters)])   
                    fExt = 'ErrorDistances.{}.{}.{}.Power.{}.h5'.format(m,h,dT,p)
                    inFile = ''.join([mDir,fExt])
                    
                    fRead = h5py.File(inFile,mode='r')
                    k = fRead.keys()[0]
                    data = np.asarray(fRead[k]).squeeze()
                    df.append(data)

                outExt = 'ErrorDistances.{}.{}.train.{}.Power.{}.csv'.format(h,m,dT,p)
                
                DF = np.concatenate(df)
                outFile = h5py.File(''.join([inDir,'ErrorDistances/',outExt]),mode='a')
                outFile.create_dataset('distances',DF)
                outFile.close()