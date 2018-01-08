#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:10:40 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.insert(0,'../../io/')
sys.path.insert(1,'../../metrics/')

import loaded as ld
import tpdMath as tm
import pandas as pd

"""
Computes the topological distance for a set of cortical parcels
"""

parser = argparse.ArgumentParser()

parser.add_argument('-td','--topoDir',help='Director where topological matrices extist.',
                    required=True,type=str,nargs='+')
parser.add_argument('-te','--topoExt',help='Extension of topological matrix files.',
                    required=True,type=str,nargs='+')
parser.add_argument('--subjectList',help='File with list of subjects.',
                    required=True,type=str)
parser.add_argument('-out','--output',help='Name of output Pandas data frame.',
                    required=True,type=str)

args = parser.parse_args()

outputName = args.output

topoDir = args.topoDir
topoExt = args.topoExt

with open(args.subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

assert len(topoDir) == 2
assert len(topoExt) == 2

for d in topoDir:
    assert os.path.exists(d)
    
df = pd.DataFrame(columns=['tpd'])
tpd = []
    
for s in subjects:
    
    v1 = ''.join([topoDir[0],s,topoExt[0]])
    v2 = ''.join([topoDir[1],s,topoExt[1]])
    
    if os.path.exists(v1) and os.path.exists(v2):
        
        v1 = ld.loadMat(v1)
        v1 = tm.tpdVector(v1)
        
        v2 = ld.loadMat(v2)
        v2 = tm.tpdVector(v2)
        
        metric = tm.tpd(v1,v2)
        tpd.append(metric)

df['tpd'] = tpd
df.to_csv(outputName)