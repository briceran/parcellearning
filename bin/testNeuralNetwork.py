import argparse
import sys
sys.path.append('..')

import parcellearning.classifierUtilities as pcu
import parcellearning.NeuralNetworkUtilities as nnu

from keras.models import load_model

import nibabel as nb
import numpy as np
import os
import pickle

# Parse the input parameters
parser = argparse.ArgumentParser(Description='Prediction using trained neural networks.')

parser.add_argument('--directory',help='Data directory.',type=str,required=True)
parser.add_argument('--datatype',help='Type of input data.',
                    choices=['Full','RestingState','ProbTrackX2'],type=str,
                    required=True)
parser.add_argument('--test',help='List of test subjects.',type=str,required=True)
parser.add_argument('--hemisphere',help='Hemisphere to proces.',
                    type=str,required=True)
parser.add_argument('--extension',help='Output directory and extension (string, separate by comma)',
                    type=str,required=True)
parser.add_argument('--power',help='Power to raise matching matrix to.',
                    type=float,required=False)

parser.add_argument('--downsample',help='Type of downsampling to perform.',default='core',
                    choices=['none','equal','core'],required=False)

# Parameters for network architecture
parser.add_argument('--layers', help='Layers in network.',type=int,required=True)
parser.add_argument('--nodes',help='Nodes per layer.',type=int,required=True)
parser.add_argument('--epochs',help='Number of epochs.',default=50,
                    type=int,required=False)
parser.add_argument('--batch',help='Batch size.',default=256,
                    type=int,required=False)

# Parameters for weight updates
parser.add_argument('--optimizer',help='Optimization.',type=str,default='rmsprop',
                    choices=['rmsprop','sgd'],required=False)
parser.add_argument('--rate',help='Learning rate.',default=0.001,
                    type=float,required=False)

args = parser.parse_args()


# Process the input arguments
try:
    testList = pcu.loadList(args.test)
except:
    raise IOError


# Data-specific parameters
dr = args.directory
dt = args.datatype
hm = args.hemisphere
sp = args.sampling
opt = args.optimizer
ext = args.extension

# Network architecture parameters
ly = args.layers
ep = args.epochs
bt = args.batchSize
rt = args.rate
nd = args.nodes

pw = args.power


prepBase = 'Prepared.{}.{}.{}.p'.format(hm,dt,ext)
prep = ''.join([dr,'Models/TestReTest/',prepBase])

print prep

mxly = 'Layers.{}.Nodes.{}'.format(ly,nd)
mxep = 'Epochs.{}.Batch.{}.Rate.{}'.format(ep,bt,rt)
mxopt = 'optimizer.{}'.format(opt)
mxt = '{}.{}'.format(dt,ext)

modelBase = 'NeuralNetwork.{}.{}.{}.{}.h5'.format(hm,mxly.lower(),mxep.lower(),mxopt.lower(),mxt)
model = ''.join([dr,'Models/TestReTest/',modelBase])

print model

assert os.path.isfile(prep)
assert os.ath.isfile(model)

"""
with open(prep,'r') as inP:
    P = pickle.load(inP)
model = load_model(model)
"""

for ts in testList:
    
    print 'Test Subject: {}'.format(ts)
    
    od = '{}Predictions/TestReTest/NeuralNetwork/{}/'.format(dr,dt)
    mxSamps = '{}.Sampling.{}.{}.Freq.{}.{}'.format(mxly,sp,mxep,pw,mxt)
    
    inFunc = '{}MyelinDensity/{}.{}.MyelinMap.32k_fs_LR.func.gii'.format(od,ts,hm)
    outPre = '{}{}.{}.{}.func.gii'.format(od,ts,hm,mxSamps)
    
    print inFunc
    print outPre
    
    """
    assert os.path.isfile(inFunc)
    
    myl = nb.load(inFunc)

    [data,match,ltvm] = P.testing(ts)
    [bl,th,pr] = nnu.predict(data,match,model,power=pw)
    
    myl.darrays[0].data = pr.astype(np.float32)
    nb.save(myl,outPre)
    """
    
    