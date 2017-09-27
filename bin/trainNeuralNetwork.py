#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:41:52 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')

import numpy as np

import parcellearning.classifierUtilities as pcu
import parcellearning.dataUtilities as pdu
import parcellearning.classifierData as pcd
import parcellearning.downsampling as pds

import parcellearning.NeuralNetwork as pNN

def buildDataMap(dataDir):
    
    """
    Method to construct data map for loading of training data.
    """
    
    dataMap = {}
    dataMap['object'] = {'{}TrainingObjects/FreeSurfer/'.format(dataDir) : 
        'TrainingObject.aparc.a2009s.h5'}
    dataMap['midline'] = {'{}Midlines/'.format(dataDir) : 
        'Midline_Indices.mat'}
    dataMap['matching'] = {'{}MatchingLibraries/Test/MatchingMatrices/'.format(dataDir) : 
        'MatchingMatrix.0.05.Frequencies.mat'}
        
    return dataMap

def downsample(inputData,method,L = None):
    
    """
    Wrapper to downsample training data.
    """
    
    methodFuncs = {'equal': pds.byMinimum,
                   'core': pds.byCore}
    
    if not L:
        L = np.arange(1,181)
    else:
        L = np.arange(1,L+1)

    x = inputData[0]
    y = inputData[1]
    m = inputData[2]
    
    [x,y,m] = methodFuncs[method](x,y,m,L)
    
    x = pdu.mergeValueArrays(x)
    y = pdu.mergeValueLists(y)
    m = pdu.mergeValueArrays(m)
    
    return [x,y,m]

parser = argparse.ArgumentParser()

# Parameters for input data
parser.add_argument('--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('--features',help='Features to include in model.',required=True)
parser.add_argument('--training',help='Subjects to train model on.',required=True)
parser.add_argument('--hemisphere',help='Hemisphere to proces.',required=True)
parser.add_argument('--output',help='Name of file storing training model',required=True)

parser.add_argument('--testing',help='Subject to test model on.',required=False)
parser.add_argument('--downsample',help='Type of downsampling to perform.',default='none',
                    choices=['none','equal','core'],required=False)

parser.add_argument('--eval',help='Evaluation dataset size.',
                    type=float,default=0.1,required=False)


# Parameters for network architecture
parser.add_argument('--layers', help='Layers in network.',type=int,required=False)
parser.add_argument('--nodes',help='Nodes per layer.',type=int,required=False)
parser.add_argument('--epochs',help='Number of epochs.',type=int,required=False)
parser.add_argument('--batch',help='Batch size.',type=int,required=False)

# Parameters for weight updates
parser.add_argument('--optimizer',help='Optimization.',type=str,
                    choices=['rmsprop','sgd'],required=False)
parser.add_argument('--rate',help='Learning rate.',type=float,
                    required=False)

args = parser.parse_args()
params = vars(args)



try:
    trainList = pcu.loadList(args.training)
except:
    raise IOError('Training list does not exist.')

try:
    testList = pcu.loadList(args.testing)
except:
    pass

dataMap = buildDataMap(args.dataDirectory)
features = args.features.split(',')

P = pcd.Prepare(dataMap,args.hemisphere,features)
trainData = P.training(trainList)
[trainData,valData] = pcu.validation(trainData,args.eval)

if args.downsample:
    trainData = downsample(trainData,args.downsample)

trainData = pcu.shuffle(trainData)
    
N = pNN.Network()
N.set_params(**params)


