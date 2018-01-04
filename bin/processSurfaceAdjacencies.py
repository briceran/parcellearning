#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:03:34 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.insert(0, '../../shortestpath/shortestpath/')

import Adjacency as aj
import json

parser = argparse.ArgumentParser()
parser.add_argument('-sd','--surfaceDir',help='Directory where surfaces exist.',
                    required=True,type=str)
parser.add_argument('-se','--surfaceExt',help='Surface file extension.',
                    required=True,type=str)
parser.add_argument('-od','--outDir',help='Output directory.',
                    required=True,type=str)
parser.add_argument('-oe','--outExt',help='Output extension.',
                    required=True,type=str)
parser.add_argument('--subjectList',help='Full path to list of subjects.',
                    required=True,type=str)
args = parser.parse_args()

sd = args.surfaceDir
se = args.surfaceExt

od = args.outDir
oe = args.outExt

if not os.path.exists(od):
    os.makedirs(od)

subjectList = args.subjectList
with open(subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

for subj in subjects:
    
    inSurf = ''.join([sd,subj,se])
    outAdjList = ''.join([od,subj,oe])
    
    print inSurf
    print outAdjList
    
    if os.path.exists(inSurf) and not os.path.exists(outAdjList):
        
        S = aj.SurfaceAdjacency(inSurf)
        S.generate_adjList()
        
        strAdj = {str(k): map(str,S.adj_list[k]) for k in S.adj_list.keys()}
        
        with open(outAdjList,'r') as outJ:
            json.dump(strAdj,outJ)
            