#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:59:08 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.append('..')
sys.path.insert(0, '../../tms/tms/')

import numpy as np
import scipy.io as sio

import json
import parcellearning.loaded as ld
import topologyVertex as tv

parser = argparse.ArgumentParser()
parser.add_argument('-ld','--labelDir',help='Directory where labels exist.',
                    required=True,type=str)
parser.add_argument('-le','--labelExt',help='Label file extension.',required=True,
                    type=str)

parser.add_argument('-sj','--surfAdj',help='Directory where adjacency lists '
                    'exist.',required=True,type=str)
parser.add_argument('-se','--surfExt',help='Extension of adjacency list file.',
                    required=True,type=str)

parser.add_argument('-od','--outDir',hepl='Output directory for topology vectors.',
                    required=True,type=str)
parser.add_argument('-oe','--outExt',hepl='Toplogy vector file extension.',
                    required=True,type=str)

parser.add_argument('-mv','--maxLabel',help='Max label value to consider.',
                    required=True,type=int)

parser.add_argument('--subjectList',help='Full path to list of subjects.',
                    required=True,type=str)

args = parser.parse_args()

lbd = args.labelDir
le = args.labelExt

sj = args.surfAdj
se = args.surfExt

od = args.outDir
oe = args.outExt

if not os.path.exists(od):
    os.makedirs(od)

mv = args.maxLabel

with open(args.subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

values = np.arange(1,mv+1)

for subj in subjects:
    
    labelFile = ''.join([lbd,subj,le])
    surfAdj = ''.join([sj,subj,se])
    outFile = ''.join([od,subj,se])
    
    if os.path.exists(labelFile) and os.path.exists(surfAdj):
        if not os.path.exists(outFile):
            
            label = ld.loadGii(labelFile,darrays=0)
            
            with open(surfAdj,'r') as inJ:
                adj = json.load(inJ)
            adj = {int(k): map(int,adj[k]) for k in adj.keys()}
            
            # Compute the topology structure of a given label map
            # topMat is a row-normalized data matrix
            _,topMat = tv.labelCounts(label,values,adj)
            
            tm = {'topology': topMat}
            sio.savemat(outFile,tm)
            
            