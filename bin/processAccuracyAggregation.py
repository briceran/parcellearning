#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 00:33:10 2018

@author: kristianeschenburg
"""

import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    baseDirectory = '/mnt/parcellator/parcellation/parcellearning/Data/Predictions/'
    iters = np.arange(5)
    
    models = ['GMM','RandomForest','NeuralNetwork']
    hemis = ['L','R']
    power = [0,1.0]
    
    datas = ['Full','RestingState','ProbTrackX2']
    dataExts = ['Combined','RestingState','Structural']
    
    dataMap = dict(zip(datas,dataExts))
    
    cols = ['model','hemisphere','power','data','accuracy']
    df = pd.DataFrame(columns = cols)
    
    for it in iters:
        
        iter_dir = ''.join([baseDirectory,'Model_{}/Accuracy/'.format(it)])
        
        for m in models:
            for h in hemis:
                for p in power:
                    for d in datas:
                        
                        baseExt = '.Model_{}.Accuracy.csv'.format(it)
                        baseName = ''.join([h,'.',m,'.',d,'.Power.',str(p),baseExt])
                        inFile = ''.join([iter_dir,baseName])

                        header = [m,h,p,d]

                        if os.path.exists(inFile):
                            
                            tempDF = pd.read_csv(inFile)
                            
                            for a in tempDF['accuracy']:
                                header = header + [a]
                                df.append(dict(zip(cols,header)),ignore_index=True)

df.to_csv(''.join([baseDirectorypa,'Accuracy.csv']))
                
        