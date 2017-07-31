#!/bin/bash

PYTHON=/project_space/parcellation/Software/anaconda2/bin/python

kind=$1
hemisphere=$2

# Define directories
dataDir=/mnt/parcellator/parcellation/parcellearning/Data/
binDir=/mnt/parcellator/parcellation/GitHub/parcellearning/bin/
script=${binDir}neuralNetworks.py

outDir=${dataDir}Models/

if [ ! -d "$outDir" ]; then
	mkdir ${outDir}
fi

N=9

if [ $kind = "ptx" ]; then
	exten="ProbTrackX2"
	feats="pt_cort,pt_subcort,sulcal,myelin,curv,label"
elif [ $kind = "fs" ]; then
	exten="RestingState"
	feats="fs_cort,fs_subcort,sulcal,myelin,curv,label"
elif [ $kind = "full" ]; then
	exten="Full"
	feats="fs_cort,fs_subcort,pt_cort,pt_subcort,sulcal,myelin,curv,label"
fi

layers=3
nodes=1250
downSample='equal'
epochs=60
batchSize=256
rate=0.001

if [ $hemisphere = 'Left' ]; then
	H='L'
elif [ $hemisphere = 'Right' ]; then
	H='R'
fi

outFileExtension=NeuralNetwork.${H}.Layers.${layers}.Nodes.${nodes}.Sampling.${downSample}.Epochs.${epochs}.Batch.${batchSize}.Rate.${rate}.${exten}

echo ${outFileExtension}

for i in $(seq 0 $N); do
	outFile=${outDir}${outFileExtension}.Iteration_${i}.p
	trainingList=${dataDir}TrainingSubjects.${i}.txt
<<<<<<< HEAD
	logFile=${outDir}logFile.${i}.${exten}.${H}.${i}.out
	if [ ! -f ${outFile}.h5 ]; then
		nohup ${PYTHON} ${script} -dDir ${dataDir} -f ${feats} -sl ${trainingList} -hm ${hemisphere} -o ${outFile} -ds ${downSample} -l ${layers} -n ${nodes} -e ${epochs} -b ${batchSize} -r ${rate} >& ${logFile} 2>&1&
	fi
done
=======
	logFile=${outDir}logFile.NeuralNetwork.${exten}.${H}.${i}.out
	nohup ${PYTHON} ${script} -dDir ${dataDir} -f ${feats} -sl ${trainingList} -hm ${hemisphere} -o ${outFile} -ds ${downSample} -l ${layers} -n ${nodes} -e ${epochs} -b ${batchSize} -r ${rate} >& ${logFile} 2>&1&
done

>>>>>>> d75d2074389929b8ab00e8eef817887334348cf5
