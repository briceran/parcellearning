#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:01:20 2017

@author: kristianeschenburg
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:15:17 2017

@author: kristianeschenburg
"""

import argparse

import sys
sys.path.append('..')

import parcellearning.MixtureModel as MM

import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Compute random forest predictions.')

# Parameters for input data
parser.add_argument('-dDir','--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('-f','--features',help='Features to include in model.',required=True)
parser.add_argument('-sl','--subjectList',help='List of subjects to include.',required=True)
parser.add_argument('-hm','--hemisphere',help='Hemisphere to proces.',required=True)
parser.add_argument('-o','--output',help='Name of file storing training model',required=True)

parser.add_argument('-cov','--covariance',help='Type of covariance matrix.',
                    default='diag',required=False)
parser.add_argument('-nc','--components',help='Number of components per model.',type=int,
                    default=2,required=False)

args = parser.parse_args()

dataDir = args.dataDirectory
features = list(args.features.split(','))
subjectList = args.subjectList
hemi = args.hemisphere
outFile = args.output

covariance = args.covariance
components = args.components

if args.numAtlases is None:
    numAtlases = args.numAtlases
else:
    numAtlases = int(args.numAtlases)

with open(subjectList,'r') as inSubjects:
    subjects = inSubjects.readlines()
subjects = [x.strip() for x in subjects]

# Generate parameter dictionary to update base atlases
parameters = ['covariance_type','n_components']
kars = {}.fromkeys(parameters)
kars['covariance_type'] = covariance
kars['n_components'] = components


print '\nParameters of mixture model:'
print '\t  components per mixture: {}'.format(components)
print '\t covariance type: {}'.format(covariance)
print '\t hemisphere: {}\n'.format(hemi)

G = MM.GMM()
tr,re = G.loadTraining(subjects,dataDir,hemi,features)
G.fit(tr,**kars)

with open(outFile,'w') as outGMM:
    pickle.dump(G,outGMM,-1)