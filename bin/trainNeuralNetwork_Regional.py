#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:46:23 2017

@author: kristianeschenburg
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:41:52 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')

import os
import pickle

import parcellearning.classifierUtilities as pcu
import parcellearning.classifierData as pcld
import parcellearning.NeuralNetwork as pNN


def save(modelDir,modelName,network):
    
    """
    Wrapper to save model and model history.
    """

    modelHistory =  network.history.history
    te = network.teConstraint
    tr = network.trConstraint
    
    full = dict(modelHistory.items() + te.metrics.items() + tr.metrics.items())

    outNetwk = ''.join([modelDir,modelName,'.h5'])
    outHistr = ''.join([modelDir,modelName,'.History.p'])
    print outNetwk
    print outHistr 
    network.model.save(outNetwk)
    
    with open(outHistr,'w') as outH:
        pickle.dump(full,outH,-1)

parser = argparse.ArgumentParser()

# Parameters for input data
parser.add_argument('--directory',help='Directory where data exists.',
                    type=str,required=True)
parser.add_argument('--datatype',help='Type of input data.',
                    choices=['Full','RestingState','ProbTrackX2'],type=str,
                    required=True)
parser.add_argument('--features',help='Features to include in model.',nargs='+',
                    type=str,required=True)

parser.add_argument('--train',help='Subjects to train model on.',
                    type=str,required=True)
parser.add_argument('--hemisphere',help='Hemisphere to proces.',
                    type=str,required=True)
parser.add_argument('--extension',help='Output directory and extension (string, separate by comma)',
                    type=str,required=True)

parser.add_argument('-mld','--modelDirectory',help='Path to specific model directory.',
                    required=True,type=str)
parser.add_argument('-trd','--trainObjectDirectory',help='Path to specific training object direcotry.',
                    required=True,type=str)
parser.add_argument('-tre','--trainObjectExtension',help='Training object extension.',
                    required=True,type=str)

parser.add_argument('--downsample',help='Type of downsampling to perform.',default='core',
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
params = {k: v for k,v in params.items() if v}



# Get list of training subjects
assert os.path.isfile(args.train)

try:
    trainList = pcu.loadList(args.train)
except:
    raise IOError('Training list does not exist.')

# Get directory where training objects exist
if args.trainObjectDirectory:
	trd = args.trainObjectDirectory
else:
	trd = None

# Get training object extension
if args.trainObjectExtension:
	tre = args.trainObjectExtension
else:
	tre = None
	
# Building training object data map
dataMap = pcu.buildDataMap(args.directory,trd,tre)

# Get names of features
features = args.features
# Get hemisphere
hemi = args.hemisphere
# Get type of data to process (resting state, full, probtrackx2)
datatype = args.datatype

# 
if not args.modelDirectory:
    modelSubDir = 'Models/TestReTest/'
else:
    modelSubDir = args.modelDirectory

modelDir = ''.join([args.directory,modelSubDir])
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)
extension = args.extension.split(',')
outDir = extension[0]
outExt = extension[1]

# Load the training data
P = pcld.Prepare(dataMap,hemi,features)
print vars(P)
print trainList
trainData = P.training(trainList)

# Save the Prepare object -- contains scaling transformation for new subjects
# as well as feature names for loading
outPrep = '{}Prepared.{}.{}.{}.p'.format(modelDir,hemi,datatype,outExt)
if not os.path.isfile(outPrep):
    with open(outPrep,'w') as outP:
        pickle.dump(P,outP,-1)
        
# Generate training and validation data
[trainData,valData] = pcu.validation(trainData,args.eval)

# Downsample the training data
if args.downsample:
    trainData = pcu.downsample(trainData,args.downsample)

# Shuffle the training data
trainData = pcu.shuffle(trainData)
    
# Construct network and update parameters
N = pNN.Network()
N.set_params(**params)

# Fit model
N.fit(trainData,valData)

# Save model
prefix = ''.join([outDir,'NeuralNetwork.{}'.format(hemi)])
suffix = ''.join([datatype,'.{}'.format(outExt)])
print prefix
print suffix
save(prefix,suffix,N)
