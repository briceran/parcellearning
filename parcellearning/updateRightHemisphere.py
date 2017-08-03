#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 19:50:05 2017

@author: kristianeschenburg
"""

import os
import h5py

dataDir = '/mnt/parcellator/parcellation/parcellearning/Data/TrainingObjects/FreeSurfer/'
subjectList = '/mnt/parcellator/parcellation/HCP/Connectome_4/SubjectList.txt'

with open(subjectList,'r') as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

for s in subjects:
    sData = '{}{}.R.TrainingObject.aparc.a2009s.h5'.format(dataDir,s)
    
    if os.path.isfile(sData):
        data = h5py.File(sData,mode='r+')
        
        dset = data[data.keys()[0]]
        
        for feature in dset.keys():
            tempData = dset[feature]
            
            if tempData.ndim == 1:
                tempData.shape+=(1,)
                del(dset[feature])
                dset.create_dataset(feature,tempData)
        print data[data.keys()[0]].keys()
        data.close()
    