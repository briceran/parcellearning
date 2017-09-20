import argparse
import sys
sys.path.append('..')

import parcellearning.classifierData as cld
import parcellearing.classifierUtilities as clu

import loaded as ld

from keras.models import load_model

import glob
import os
import pickle
import nibabel as nb
import numpy as np

# Parse the input parameters
parser = argparse.ArgumentParser(Description='Prediction using trained neural networks.')
parser.add_argument('--hemisphere',help='Hemisphere of the training data.',
	required=True,type=str)
parser.add_argument('--sampling',help='Type of downsample that was performed.',
	required=False,default='equal',type=str)

parser.add_argument('--layers',help='Number of layers in model.',
	required=True,type=int)
parser.add_argument('--nodes',help='Number of nodes per layer.',
	required=True,type=int)
parser.add_argument('--epochs',help='Number of training epochs.',
	equired=True,type=int)
parser.add_argument('--batchSize',help='Size of training batches.',
	required=True,type=int)
parser.add_argument('--rate',help='Training rate.',
	required=True,type=float)
parser.add_argument('--power',help='Power of mapping frequencies.',
	required=False,default=1,type=int)

parser.add_argument('--training',help='Training subject file.',
	required=True,type=str)
parser.add_argument('--testing',help='Testing subject file.',
	required=True,type=str)
parser.add_argument('--baseDirectory',help='Base directory where data exists.',
	required=True,type=str)
parser.add_argument('--extension',help='Model file extension.',
	required=True,type=str)

args = parser.parse_args()

# Process the input arguments

# Data-specific parameters
hemisphere = args.hemisphere
sampling = args.sampling
trainFile = args.training
testFile = args.testing
baseDir = args.baseDirectory
extension = args.extension

# Network architecture parameters
layers = args.layers
epochs = args.epochs
batchSize = args.batchSize
rate = args.rate
power = args.power
nodes = args.nodes


# Dictionary with file directories and extensions
dataMap = {'object': {'{}TrainingObjects/FreeSurfer/'.format(baseDir) :
						'TrainingObject.aparc.a2009s.h5'},
			'matching' : {'{}MatchingLibraries/Test/MatchingMatrices/'.format(baseDir) :
						'MatchingMatrix.0.05.Frequencies.mat'},
			'midline' : {'{}Midlines/'.format(baseDir) :
						'Midline_Indices.mat'}
			}

# Dictionary mapping data type to feature types
featureMap = {'RestingState': ['fs_cort','fs_subcort','sulcal','myelin','curv'],
				'ProbTrackX2' : ['pt_cort','pt_subcort','sulcal','myelin','curv'],
				'Full' : ['fs_cort','fs_subcort','pt_cort','pt_subcort','sulcal','myelin','curv']}


# Load training subjects
with open(trainFile,'r') as inTrain:
	trainSubjects = inTrain.readlines()
trainSubjects = [x.strip() for x in trainSubjects]

with open(testFile,'r') as inTest:
	testSubjects = inTest.readlines()
testSubjects = [x.strip() for x in testSubjects]

dataTypes = ['RestingState','ProbTrackX2','Full']

for D in dataTypes:

	modelExtension = '{}.{}.h5'.format(D,extension)
	modelPrefix = 'NeuralNetwork'

	features = featureMap[D]
	modelBase = '{}.{}.Layers.{}.Nodes.{}.Sampling.{}.Epochs.{}.Batch.{}.Rate.{}.{}'.format(modelPrefix,
		hemisphere,layers,nodes,sampling,epochs,batchSize,rate,modelExtension)
	modelFile = '{}Models/{}'.format(baseDir,modelBase)

	print modelFile
	print os.path.isfile(modelFile)

	model = load_model(modelFile)

	P = cld.Prepare(dataMap,hemisphere,features)
	[data,labels,matches] = P.training(trainSubjects)

	for test_subj in testSubjects:

		[x_test,matchingMatrix,ltvm] = P.testing(test_subj)
		[baseline,thresholded,prediction] = model.predict(x_test)


