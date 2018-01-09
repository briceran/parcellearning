import argparse
import sys
sys.path.append('..')

import h5py
import numpy as np
import os

import parcellearning.loaded as ld

parser = argparse.ArgumentParser(description='Build training objects.')
parser.add_argument('--directory',help='Base directory where data exists.',type=str)
parser.add_argument('--hemisphere',help='hemisphere to process.',type=str,choices=['L','R'])
parser.add_argument('--subjectID',help='Name of subject for which to build training object.',type=str)
parser.add_argument('--outputFile',help='Name training object file.',type=str)

parser.add_argument('-fn','--featureName',help='Feature name to include in training object.',
	action='append',type=str)
parser.add_argument('-fd','--featureDirectory',help='Sub-directory where feature data exists.',
	action='append',type=str)
parser.add_argument('-fe','--featureExtension',help='Extension of file to include in training object.',
	action='append',type=str)

args = parser.parse_args()

directory = args.directory
hemisphere = args.hemisphere
output = args.outputFile
subjectID = args.subjectID

featName = args.featureName
featDirs = args.featureDirectory
featExts = args.featureExtension

assert os.path.isdir(directory)
assert len(featName) == len(featDirs) == len(featExts)

dataObject = h5py.File(output,mode='w')
dataObject.create_group(subjectID)
dataObject.attrs['ID'] = subjectID

loadingFunctions = {'gii': ld.loadGii,
					'mat': ld.loadMat,
					'p': ld.loadPick}

for i,feat in enumerate(featName):

	featExtension = featExts[i]
	featDirectory = featDirs[i]
	extension = ''.join([subjectID,'.',hemisphere,'.',featExtension])

	featFile = ''.join([directory,featDirectory,extension])

	print featFile
	assert os.path.isfile(featFile)
	fileParts = featFile.split('.')

	featureData = loadingFunctions[fileParts[-1]](featFile)
	if featureData.ndim == 1:
            featureData.shape+=(1,)

	dataObject[subjectID].create_dataset(feat,data=featureData)
	print '{} shape: {}'.format(feat,dataObject[subjectID][feat].shape)

dataObject.close()