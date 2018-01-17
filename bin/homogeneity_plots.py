#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:27:45 2018

@author: kristianeschenburg
"""

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

hemi = ['L','R']
models = ['GMM','NeuralNetwork','RandomForest']
data = ['ProbTrackX2','RestingState','Full']

inDir = '/Users/kristianeschenburg/Documents/Research/IBIC/SPIE_2018/Results/Homogeneity/'

fullData = []
mu = []

darrays = []

d0s = []
d1s = []

GMM_0 = []
GMM_1 = []

NN_0 = []
NN_1 = []

RF_0 = []
RF_1 = []

D = {'GMM,0': GMM_0,
     'GMM,1': GMM_1,
     
     'NeuralNetwork,0': NN_0,
     'NeuralNetwork,1': NN_1,
     
     'RandomForest,0':RF_0,
     'RandomForest,1':RF_1}

for m in models:
    for dT in data:
        for h in hemi:
            
            STR = ','.join([m,dT])

            
            ptx1 = ''.join([inDir,'Homogeneity.{}.{}.train.{}.hmg.ProbTrackX2.Power.0.csv'.format(h,m,dT)])
            rest1 = ''.join([inDir,'Homogeneity.{}.{}.train.{}.hmg.RestingState.Power.0.csv'.format(h,m,dT)])
            comb1 = ''.join([inDir,'Homogeneity.{}.{}.train.{}.hmg.Full.Power.0.csv'.format(h,m,dT)])
            
            ptx2 = ''.join([inDir,'Homogeneity.{}.{}.train.{}.hmg.ProbTrackX2.Power.1.csv'.format(h,m,dT)])
            rest2 = ''.join([inDir,'Homogeneity.{}.{}.train.{}.hmg.RestingState.Power.1.csv'.format(h,m,dT)])
            comb2 = ''.join([inDir,'Homogeneity.{}.{}.train.{}.hmg.Full.Power.1.csv'.format(h,m,dT)])
            
            p0 = [ptx1,rest1,comb1]
            p1 = [ptx2,rest2,comb2]
            
            for k in np.arange(3):

                print m,dT,h,data[k]

                d0 = p0[k]
                data0 = pd.read_csv(d0)
                data0 = data0[data0.columns[2:]]
                D0 = np.asarray(data0)

                d = {'Mean': np.nanmean(D0),
                     'Training': dT,
                     'Power': 0,
                     'Model': m,
                     'Homogen':data[k],
                     'Hemisphere': h}
                
                mu0 = pd.DataFrame(columns=['Mean','Training','Power','Model','Homogen','Hemisphere'])
                mu0 = mu0.append(d,ignore_index=True)
                
                d1 = p1[k]
                data1 = pd.read_csv(d1)
                data1 = data1[data1.columns[2:]]
                D1 = np.asarray(data1)
                
                print 'Power: 0   ',np.nanmean(D0)
                print 'Power: 1   ',np.nanmean(D1)

                d = {'Mean': np.nanmean(D1),
                     'Training': dT,
                     'Power': 1,
                     'Model': m,
                     'Homogen': data[k],
                     'Hemisphere': h}
                mu1 = pd.DataFrame(columns=['Mean','Training','Power','Model','Homogen','Hemisphere'])
                mu1 = mu1.append(d,ignore_index=True)
            
                mu.append(mu0)
                s0 = ','.join([m,'0'])
                D[s0].append(D0)
                mu.append(mu1)
                s1 = ','.join([m,'1'])
                D[s1].append(D1)
                
                darrays.append(D0)
                darrays.append(D1)

            """
            e = sns.factorplot(x="Training", y="Mean", hue="Power",
                               data=fullData,kind="bar", palette="muted",
                               legend=False,size=8);
            
            e.despine(left=True);
            plt.ylim(0,1);
            plt.ylabel('Mean Homogeneity',fontsize=20);
            plt.xlabel('Data Type',fontsize=20);
            plt.yticks(fontsize=18);
            plt.xticks(fontsize=18);
            plt.title('Model: GMM \n Hemisphere: L',fontsize=20)
            plt.text(2.32,0.83,'a',fontdict=None,fontsize=30)
            legend=plt.legend(loc='best',title='Power')
            legend.get_title().set_fontsize('20')
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
            plt.tight_layout()
            plt.savefig('/Users/kristianeschenburg/Desktop/GMM.Homogeneity.Barplot.jpg')
            """

fullData = pd.concat(mu)