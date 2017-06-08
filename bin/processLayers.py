#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:54:09 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')

import parcellearning.loaded as ld
import parcellearning.regionalizationMethods as RM

import os
import pickle

__author__ = 'kristianeschenburg'

parser = argparse.ArgumentParser(description='This function computes the level structures of a cortical map file.')

parser.add_argument('-sl','--subjectList', help='Name of file containing subject IDs.',required=True)
parser.add_argument('-surfAdj','--surfaceAdjacency', help='Path to surface adjacency file.',required=True)

parser.add_argument('-lDir','--labelDir',help='Path to label files.',required=True)
parser.add_argument('-lExt','--labelExt',help='Extension of label files.',required=True)

parser.add_argument('-bDir','--boundaryDir',help='Path to boundary vertex files.',required=True)
parser.add_argument('-bExt','--boundaryExt',help='Boundary vertex file extension.',required=True)

parser.add_argument('-oDir','--outputDir',help='Output directory for level structures.',required=True)
parser.add_argument('-oExt','--outputExt',help='Output level structure file extension.',required=True)

args = parser.parse_args()

subjectList = args.subjectList
surfAdjFile = ''.join([args.surfaceAdjacency])


print('Loading subject list.')
subjectList = ''.join([subjectList])

with open(subjectList,"rb") as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]


labelDir = args.labelDir
labelExt = args.labelExt

boundaryDir = args.boundaryDir
boundaryExt = args.boundaryExt

outDir = args.outputDir
outExt = args.outputExt


for s in subjects:
    cond = True
    
    inLabel = ''.join([labelDir,s,labelExt])
    print(inLabel)
    inBound = ''.join([boundaryDir,s,boundaryExt])
    print(inBound)
    outFile = ''.join([outDir,s,outExt])
    
    if not os.path.isfile(inLabel):
        cond = False
        print 'Label file for {} does not exist.'.format(s)
        
    if not os.path.isfile(inBound):
        cond = False
        print 'Boundary file for {} does not exist.'.format(s)
    
    if cond:
        
        print 'Computing level structures of {}.'.format(s)
        LAY = RM.computeLabelLayers(labelFile = inLabel,
                                    surfaceAdjacency = surfAdjFile,
                                    borderFile = inBound)
        
        with open(outFile,"wb") as output:
            pickle.dump(LAY,output,-1)

