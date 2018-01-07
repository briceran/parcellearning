#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:41:34 2018

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.insert(0,'../../metrics/')
sys.path.insert(1,'../../io/')

import distanceMetrics as distMet
import json
import loaded as ld
import networkx as nx

parser = argparse.ArgumentParser()

parser.add_argument('-sm','--scalarMap',help='Scalar map used to compute similarity.',
                    required=True,type=str)
parser.add_argument('-sf','--surfAdj',help='Surface adjacency file.',required=True,type=str)
parser.add_argument('-sp','--samples',help='Number of vertices to sample.',
                    required=True,type=int)
parser.add_argument('-md','--maxDistance',help='Maximum distance for Dijkstra.',
                    required=True,type=int)
parser.add_argument('-out','--output',help='Output file.',required=True,type=str)

args = parser.parse_args()

scalarMap = args.scalarMar
surfAdj = args.surfAdj
samples = args.samples
maxDist = args.maxDistance
output = args.output

assert os.path.exists(scalarMap)
scalarMap = ld.loadGii(scalarMap,darray=1)

assert os.path.exists(surfAdj)
with open(surfAdj,'r') as inJ:
    J = json.load(inJ)
adj = {int(k): map(int,inJ[k]) for k in inJ.keys()}
G = nx.from_dict_of_lists(adj)

assert samples > 0
assert maxDist <= len(scalarMap)

distanceMaps = distMet.scalarDistance(scalarMap,G,samples,maxDist)

with open(output,'w') as outJ:
    json.dump(distanceMaps,outJ)