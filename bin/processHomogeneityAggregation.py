#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:14:32 2018

@author: kristianeschenburg
"""

import pandas as pd
import numpy as np

power = [0,1.0]
hemi = ['L','R']
models = ['GMM','NeuralNetwork','RandomForest']
data = ['Full','RestingState','ProbTrackX2']
x = np.arange(5)

inDir = '/mnt/parcellator/parcellation/parcellearning/Data/Predictions/'

for p in power:
    for h in hemi:
        for m in models:
            for dT in data:
                for dH in data:
                    df = []
                    dfs = []
                    
                    for iters in x:
                        
                        mDir = ''.join([inDir,'Model_{}/Homogeneity/'.format(iters)])
                        fExt = 'Homogeneity.{}.{}.train.{}.hmg.{}.Power.{}.Iteration_{}.csv'.format(h,m,dT,dH,p,iters)
                        fExtS = 'Homogeneity.{}.{}.train.{}.hmg.{}.Power.{}.Iteration_{}.size.csv'.format(h,m,dT,dH,p,iters)
                        inFile = ''.join([mDir,fExt])
                        inFileS = ''.join([mDir,fExtS])
                        
                        df.append(pd.read_csv(inFile))
                        dfs.append(pd.read_csv(inFileS))
                    
                    outExt = 'Homogeneity.{}.{}.train.{}.hmg.{}.Power.{}.csv'.format(h,m,dT,dH,p)
                    outExtS = 'Homogeneity.{}.{}.train.{}.hmg.{}.Power.{}.size.csv'.format(h,m,dT,dH,p)
                    
                    DF = pd.concat(df)
                    DF.to_csv(''.join([inDir,outExt]))
                    DFS = pd.concat(dfs)
                    DFS.to_csv(''.join([inDir,outExtS]))
                        
                        
                        