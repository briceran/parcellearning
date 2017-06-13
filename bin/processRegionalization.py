#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:48:44 2017

@author: kristianeschenburg
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:54:09 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')

import parcellearning.regionalizationMethods as RM

import h5py
import os

__author__ = 'kristianeschenburg'

parser = argparse.ArgumentParser(description='This function computes the level structures of a cortical map file.')

parser.add_argument('-sl','--subjectList', help='Name of file containing subject IDs.',required=True)
parser.add_argument('-level','--layerLevel',help='Level of layers to consider.',required=True)

parser.add_argument('-restDir','--restingStateDir', help='Path to resting state files.',required=True)
parser.add_argument('-restExt','--restingStateExt',help='Resting state file extension.',required=True)

parser.add_argument('-midDir','--midlineDir',help='Path to midline indices.',required=True)
parser.add_argument('-midExt','--midlineExt',help='Midline indices file extension.',required=True)

parser.add_argument('-lyDir','--layerDir',help='Path to layer file.',required=True)
parser.add_argument('-lyExt','--layerExt',help='Layer file extension.',required=True)

parser.add_argument('-oDir','--outputDir',help='Output directory for regionalization.',required=True)
parser.add_argument('-oExt','--outputExt',help='Output regionalization file extension.',required=True)

args = parser.parse_args()

print('Loading subject list.')
subjectList = args.subjectList
subjectList = ''.join([subjectList])

with open(subjectList,"rb") as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]

restDir = args.restingStateDir
restExt = args.restingStateExt

midDir = args.midlineDir
midExt = args.midlineExt

layerDir = args.layerDir
layerExt = args.layerExt

outDir = args.outputDir
outExt = args.outputExt

level = args.layerLevel


for s in subjects:
    cond = True
    
    inLayer = ''.join([layerDir,s,layerExt])
    inRest = ''.join([restDir,s,restExt])
    inMid = ''.join([midDir,s,midExt])
    
    outFile = ''.join([outDir,s,outExt])
    
    if not os.path.isfile(inLayer):
        cond = False
        print 'Layer file for {} does not exist.'.format(s)
        
    if not os.path.isfile(inRest):
        cond = False
        print 'Resting state file for {} does not exist.'.format(s)
    
    if cond:
        
        print 'Computing level structures of {}.'.format(s)
        LAY = RM.regionalizeStructures(inRest,inLayer,level,inMid,
                                       measure='median',R=180)
        
        regFile = h5py.File(outFile,'w')
        regFile['reg'] = LAY
        regFile.close()