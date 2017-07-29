#!/bin/bash

PYTHON=/project_space/parcellation/Software/anaconda2/bin/python

kind=$1

# Define directories
dataDir=/mnt/parcellator/parcellation/parcellearning/Data/
binDir=/mnt/parcellator/parcellation/GitHub/parcellearning/bin/
script=${binDir}neuralNetworks.py

outDir=${dataDir}Models/

if [ ! -d "$outDir" ]; then
	mkdir ${outDir}
fi

N=9

inSubj=${dataDir}SmallList.txt

if [ $kind = "ptx" ]; then
	exten="ProbTracX2"
	feats="pt_cort,pt_subcort,sulcal,myelin,curv,label"
elif [ $kind = "fs" ]; then
	exten="RestingState"
	feats="fs_cort,fs_subcort,sulcal,myelin,curv,label"
elif [ $kind = "full" ]; then
	exten="Full"
	feats="fs_cort,fs_subcort,pt_cort,pt_subcort,sulcal,myelin,curv,label"
fi

layers=10
nodes=10
downSample='equal'
hemisphere='Left'
epochs=20
batchSize=256
rate=0.001

if [ $hemisphere = 'Left' ]; then
	H='L'
elif [ $hemisphere = 'Right' ]; then
	H='R'
fi

outFileExtension=${H}.Layers.${layers}.Nodes.${nodes}.Sampling.${downSample}.Epochs.${epochs}.Batch.${batchSize}.Rate.${rate}.${exten}

for i in $(seq 0 $N); do
	outFile=${outDir}NetworkModel.${outFileExtension}.Iteration_${i}
	trainingList=${dataDir}TestingSubjects.${i}.txt
	logFile=${outDir}logFile.${i}.out
	nohup ${PYTHON} ${script} -dDir ${dataDir} -f ${feats} -sl ${trainingList} -hm ${hemisphere} -o ${outFile} -ds ${downSample} -l ${layers} -n ${nodes} -e ${epochs} -b ${batchSize} -r ${rate} >& ${logFile} 2>&1&done

