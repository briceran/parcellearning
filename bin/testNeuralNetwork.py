import argparse
import sys
sys.path.append('..')

import parcellearning.loaded as ld
import parcellearning.classifierData as pcd
import parcellearning.classifierUtilities as pcu
import parcellearning.NeuralNetworkUtilities as nnu

from keras.models import load_model

import nibabel as nb
import numpy as np
import os
import pickle

# Parse the input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--directory',help='Base directory where all data exists.',type=str,required=True)

parser.add_argument('--objectDirectory',help='Directory where data objects exist.',type=str,
                    required=True)
parser.add_argument('--objectExtension',help='Object file extension.',type=str,required=True)

parser.add_argument('--modelDirectory',help='Directory where trained models exist.',type=str,required=True)
parser.add_argument('--modelExtension',help='Model file extension.',type=str,required=True)

parser.add_argument('--prepareDirectory',help='Directory where prepared objects exist.',type=str,required=True)
parser.add_argument('--prepareExtension',help='Prepared object file extension.',type=str,required=True)

parser.add_argument('--outputDirectory',help='Output data directory.',type=str,required=True)
parser.add_argument('--outputExtension',help='Output prediction file extension.',type=str,required=True)

parser.add_argument('--test',help='List of test subjects.',type=str,required=True)
parser.add_argument('--hemisphere',help='Hemisphere to proces.',type=str,required=True,
                    choices=['L','R'])

parser.add_argument('--power',help='Power to raise matching matrix to.',
                    type=float,required=False)

args = parser.parse_args()

baseDir = args.directory
hm = args.hemisphere
pw = args.power

# Process the input arguments
try:
    testList = pcu.loadList(args.test)
except:
    raise IOError


# Get testing object directories and data
objd = args.objectDirectory
obje = args.objectExtension
assert os.path.isdir(objd)

# Get model directories and data
md = args.modelDirectory
me = args.modelExtension
assert os.path.isdir(md)

# Get prepared object directories and data
prd = args.prepareDirectory
pre = args.prepareExtension
assert os.path.isdir(prd)

# Get output directory and data
outd = args.outputDirectory
oute = args.outputExtension
assert os.path.isdir(outd)


prepared = ''.join([prd,pre])
print prepared
assert os.path.isfile(prepared)
with open(prepared,'r') as inPrep:
    P = pickle.load(inPrep)


modelBase = ''.join(['NeuralNetwork','.',hm,'.',me])
model = ''.join([md,modelBase])
print model
assert os.path.isfile(model)
model = load_model(model)


# Loop over test subjects 
for test_subj in testList:
    
    print 'Test Subject: {}'.format(test_subj)

    # Load functional file for saving prediction later
    inFunc = '{}MyelinDensity/{}.{}.MyelinMap.32k_fs_LR.func.gii'.format(baseDir,test_subj,hm)
    assert os.path.isfile(inFunc)
    func = nb.load(inFunc)

    # Load midline indices
    mid = '{}Midlines/{}.{}.Midline_Indices.mat'.format(baseDir,test_subj,hm)
    assert os.path.isfile(mid)
    mid = ld.loadMat(mid)-1
    
    # Construct output file name
    outExtension = ''.join(['.',hm,'.',oute])
    outPrediction = ''.join([outd,test_subj,outExtension])

    if not os.path.isfile(outPrediction):
    
        # Scale test data, load matching frequencies, compute label-to-vertex mappings
        [data,match,ltvm] = pcd.testing(P,test_subj,trDir=objd,trExt=obje)
        # Compute prediction
        [baseline,threshold,predicted] = nnu.predict(data,match,model,power=pw)

        # Set midline predictions to 0
        predicted[mid]=0
        
        # Save result
        func.darrays[0].data = predicted.astype(np.float32)
        nb.save(func,outPrediction)
