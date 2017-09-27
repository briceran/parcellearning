#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:41:52 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')

import inspect
import pickle

import parcellearning.classifierUtilities as pcu
import parcellearning.classifierData as pcld
import parcellearning.NeuralNetwork as pNN


def save(outputDir,extension,network):
    
    """
    Wrapper to save model and model history.
    """
    
    hs = network.history.history
    te = network.teConstraint
    tr = network.trConstraint
    
    full = dict(hs.items() + te.metrics.items() + tr.metrics.items())
    
    base = '.'
    params,_,_,vals = inspect.getargspec(network.__init__)
    for (k,v) in zip(params[1:],vals):
        base = ''.join([base,'{}.{}.'.format(k,v)])
    
    model = network.model
    
    outModel = '{}{}{}.h5'.format(outputDir,base,extension)
    outHistr = '{}{}{}.History.p'.format(outputDir,base,extension)
    
    model.save(outModel)
    with open(outHistr,'w') as OH:
        pickle.dump(full,OH,-1)
    
    
    
parser = argparse.ArgumentParser()

# Parameters for input data
parser.add_argument('--dataDirectory',help='Directory where data exists.',required=True)
parser.add_argument('--features',help='Features to include in model.',required=True)
parser.add_argument('--train',help='Subjects to train model on.',required=True)
parser.add_argument('--hemisphere',help='Hemisphere to proces.',required=True)
parser.add_argument('--out',help='Output directory and extension (string, separate by comma)',
                    required=True)

parser.add_argument('--test',help='Subject to test model on.',required=False)
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

dataMap = pcu.buildDataMap(args.dataDirectory)
features = args.features.split(',')

P = pcld.Prepare(dataMap,args.hemisphere,features)
trainData = P.training(trainList)
[trainData,valData] = pcu.validation(trainData,args.eval)

if args.downsample:
    trainData = pcu.downsample(trainData,args.downsample)

trainData = pcu.shuffle(trainData)
    
N = pNN.Network()
N.set_params(**params)


