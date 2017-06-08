#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:41:11 2017

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

import parcellearning.loaded as ld
import parcellearning.regionalizationMethods as RM

import os
import pickle

__author__ = 'kristianeschenburg'

parser = argparse.ArgumentParser(description='This function computes the level structures of a cortical map file.')
parser.add_argument('-sl','--subjectList', help='Name of file containing subject IDs.',required=True)

parser.add_argument('-surfAdj','--surfaceAdjacency', help='Path to surface adjacency file.',required=True)

parser.add_argument('-lDir','--labelDir',help='Path to label files.',required=True)
parser.add_argument('-lExt','--labelExtension',help='Extension of label files.',required=True)

parser.add_argument('-oDir','--outputDirectory',help='Output directory for level structures.',required=True)
parser.add_argument('-oExt','--outputExtension',help='Output level structure file extension.',required=True)

args = parser.parse_args()


print ("Subject list: %s\n" % args.subjectList)
print ("Surface adjacency file: %s\n" % ''.join([args.surfaceAdjacency]))
print ("Label extension: %s\n" % ''.join([args.labelExtension]))
print ("Output directory: %s\n" % args.outputDirectory) 


subjectList = args.subjectList
surfAdjFile = ''.join([args.surfaceAdjacency])


print('Loading subject list.')
subjectList = ''.join([subjectList])


with open(subjectList,"rb") as inFile:
    subjects = inFile.readlines()
subjects = [x.strip() for x in subjects]


labelDir = args.labelDir
labelExtension = args.labelExtension


outDir = args.outputDirectory
outExt = args.outputExtension


for s in subjects:
    cond = True
    
    inLabel = ''.join([labelDir,s,labelExtension])
    outFile = ''.join([outDir,s,outExt])
    
    if not os.path.isfile(inLabel):
        cond = False
        print 'Label file for {} does not exist.'.format(s)
    
    if cond:
        
        print 'Computing boundary vertices of {}.'.format(s)
        BM = RM.coreBoundaryVertices(inLabel,surfAdjFile)
        
        with open(outFile,"wb") as output:
            pickle.dump(BM,output,-1)

