 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:41:34 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.append('..')
sys.path.insert(0,'../../metrics/')
sys.path.insert(1,'../../io/')

import h5py
import json

import parcellearning.dataUtilities as du

import distanceMetrics as distMet
import networkx as nx

parser = argparse.ArgumentParser()


parser.add_argument('-fd','--featureData',help='Path to feature matrix.',
                    required=True,type=str)
parser.add_argument('-fk','--featureKeys',help='Features to use for similarity.',
                    required=True,type=str,nargs='+')

parser.add_argument('-sf','--surfAdj',help='Surface adjacency file.',
                    required=True,type=str)

parser.add_argument('-sp','--samples',help='Number of vertices to sample.',
                    required=True,type=int)

parser.add_argument('-md','--maxDistance',help='Maximum distance for Dijkstra.',
                    required=True,type=int)
parser.add_argument('-out','--output',help='Output file.',required=True,type=str)

args = parser.parse_args()

featureMap = args.featureData
featureKeys = args.featureKeys

surfAdj = args.surfAdj
samples = args.samples
maxDist = args.maxDistance
output = args.output

# Load data array
assert os.path.exists(featureMap)
print 'Loading feature map {}.'.format(featureMap)

features = h5py.File(featureMap,mode='r')
fp = featureMap.split('/')[-1]
subject = fp.split('.')[0]
dataDict = features[subject]

dataArray = du.mergeValueArrays(dataDict,keys = featureKeys)
features.close()

# Load surface adjacency file and generate graph
assert os.path.exists(surfAdj)
print 'Generating surface graph.'
with open(surfAdj,'r') as inJ:
    J = json.load(inJ)
adj = {int(k): map(int,J[k]) for k in J.keys()}
G = nx.from_dict_of_lists(adj)

assert samples > 0
assert maxDist <= dataArray.shape[0]

samp,dim = dataArray.shape

print 'Computing distance metrics'

if dim == 1:
    print 'scalars'
    distanceMap = distMet.scalarDistance(dataArray,G,samples,maxDist)
elif dim > 1:
    print 'vectors'
    distanceMap = distMet.vectorDistance(dataArray,G,samples,maxDist)
else:
    raise('Data array is of dimension 0.')

with open(output,'w') as outJ:
    json.dump(distanceMap,outJ)