#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:43:41 2018

@author: kristianeschenburg
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:14:32 2018

@author: kristianeschenburg
"""

import pandas as pd
import numpy as np

power = [0,1]
hemi = ['L','R']
models = ['GMM','NeuralNetwork','RandomForest']
data = ['RestingState','ProbTrackX2']
x = np.arange(5)

inDir = '/mnt/parcellator/parcellation/parcellearning/Data/Predictions/'

for p in power:
    for h in hemi:
        for m in models:
            for dT in data:
                    df = []
                    dfs = []
                    
                    for iters in x:
                    
                        mDir = ''.join([inDir,'Model_{}/DiceMetrics/'.format(iters)])
                        fExt = 'Dice.{}.{}.{}.Power.{}.Iteration_{}.csv'.format(h,m,dT,p,iters)
                        inFile = ''.join([mDir,fExt])
                        
                        df.append(pd.read_csv(inFile))
                    
                    outExt = 'Dice.{}.{}.train.{}.Power.{}.csv'.format(h,m,dT,p)
                    
                    DF = pd.concat(df)
                    DF.to_csv(''.join([inDir,outExt]))

                        
                        