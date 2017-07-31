#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:15:17 2017

@author: kristianeschenburg
"""

import argparse
import time

import sys
sys.path.append('..')

import parcellearning.malp as malp


import h5py
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser(description='Compute random forest predictions.')

# Parameters for input data
parser.add_argument('-dDir','--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('-f','--features',help='Features to include in model.',required=True)
parser.add_argument('-sl','--subjectList',help='List of subjects to include.',required=True)
parser.add_argument('-hm','--hemisphere',help='Hemisphere to proces.',required=True)
parser.add_argument('-nm','--neighborhoodMap',help='Label adjacency file.',required=True)
parser.add_argument('-o','--output',help='Name of file storing training model',required=True)

parser.add_argument('-d','--depth',help='Depth of random forest trees.',type=int,
                    default=5,required=False)
parser.add_argument('-nEst','--estimators',help='Number of tree estimators per forest',type=int,
                    default=50,required=False)
parser.add_argument('-a','--atlasSize',help='Number of training subjects per atlas',type=int,
                    default=1,required=False)
parser.add_argument('-na','--numAtlases',help='Number of atlases to generate from list of training subjects.')

args = parser.parse_args()

dataDir = args.dataDirectory
features = list(args.features.split(','))
subjectList = args.subjectList
hemi = args.hemisphere
nHood = args.neighborhoodMap
outFile = args.output

depth = args.depth
nEstimators = args.estimators
atlasSize = args.atlasSize

if args.numAtlases is None:
    numAtlases = args.numAtlases
else:
    numAtlases = int(args.numAtlases)

with open(subjectList,'r') as inSubjects:
    subjects = inSubjects.readlines()
subjects = [x.strip() for x in subjects]

# Generate parameter dictionary to update base atlases
parameters = ['n_estimators','max_depth']
kars = {}.fromkeys(parameters)
kars['n_estimators'] = nEstimators
kars['max_depth'] = depth


print '\nParameters of multi-atlas:'
print '\t atlas size: {}'.format(atlasSize)
print '\t number of atlases: {}'.format(numAtlases)
print '\t hemisphere: {}\n'.format(hemi)

print 'Parameters of base-atlas:'
print '\t depth: {}'.format(depth)
print '\t n estimators: {}'.format(nEstimators)

M = malp.MultiAtlas(atlas_size=2,atlases=numAtlases)
M.loadTraining(subjects,dataDir,hemi,features)

fittedAtlases = malp.parallelFitting(M,nHood,features,**kars)

with open(outFile,'w') as outMALP:
    pickle.dump(fittedAtlases,outMALP,-1)