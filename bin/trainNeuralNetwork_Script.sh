#!/bin/bash

PYTHON=/project_space/parcellation/Software/anaconda2/bin/python

subjectList=$1
suffix=$2
layers=$3
nodes=$4
hemisphere=$5
kind=$6

dataDir=/mnt/parcellator/parcellation/parcellearning/Data/
binDir=/mnt/parcellator/parcellation/GitHub/parcellearning/bin/
script=${binDir}trainNeuralNetwork.py

N=9

# Check that "kind" is either probtrackx2, functional, or combined
if [ $kind != "ptx" ] && [ $kind != "fs" ] && [ $kind != "full" ]; then
	echo "Incorrect data type."
	exit
fi

# Check that hemisphere is either left or right
if [ $hemisphere != "Left" ] && [ $hemisphere != "Right" ]; then
	echo "Incorrect hemisphere."
	exit
fi

# and if output directory does not exist, created
if [ ! -d "$outDir" ]; then
	mkdir ${outDir}
fi

if [ $kind = "ptx" ]; then
	ext="ProbTrackX2"
	feats="pt_cort,pt_subcort,sulcal,myelin,curv,label"
elif [ $kind = "fs" ]; then
	ext="RestingState"
	feats="fs_cort,fs_subcort,sulcal,myelin,curv,label"
elif [ $kind = "full" ]; then
	ext="Full"
	feats="fs_cort,fs_subcort,pt_cort,pt_subcort,sulcal,myelin,curv,label"
fi

if [ $hemisphere = 'Left' ]; then
	H='L'
elif [ $hemisphere = 'Right' ]; then
	H='R'
fi

downSample='core'
logFile = logFile.${layers}.${nodes}.${H}.${kind}.${suffix}.log

nohup ${PYTHON} ${script} --directory ${dataDir} --datatype ${ext} --features ${feats} --train ${subjectList} --hemisphere ${H} --extension ${suffix} --downsample ${downSample} --layers ${layers} --nodes ${nodes} >& ${logFile} 2>&1&
