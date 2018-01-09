#!/bin/bash

PYTHON=/project_space/parcellation/Software/anaconda2/bin/python
#PYTHON=/Users/kristianeschenburg/anaconda/bin/python

kind=$1
hemisphere=$2

# Define directories
dataDir=/mnt/parcellator/parcellation/parcellearning/Data/
binDir=/mnt/parcellator/parcellation/GitHub/parcellearning/bin/
script=${binDir}neuralNetworks.py

#dataDir=/Users/kristianeschenburg/Desktop/Programming/Data/
#binDir=/Users/kristianeschenburg/Documents/GitHub/parcellearning/bin/
#script=${binDir}neuralNetworks.py

outDir=${dataDir}Models/

if [ ! -d "$outDir" ]; then
	mkdir ${outDir}
fi

N=2

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
downSample='equal'
nodes=100
epochs=30
batchSize=256
rate=0.001

if [ $hemisphere = 'Left' ]; then
	H='L'
elif [ $hemisphere = 'Right' ]; then
	H='R'
fi

for i in $(seq 1 $N); do
		outFileExtension=NeuralNetwork.${H}.Layers.${layers}.Nodes.${nodes}.Sampling.${downSample}.Epochs.${epochs}.Batch.${batchSize}.Rate.${rate}.${exten}.TestReTest
		echo ${outFileExtension}
		outFile=${outDir}${outFileExtension}.Iteration_${i}
		trainingList=${dataDir}TrainTestLists/TestRetest_Train${i}.txt
		logFile=${outDir}logFile.NeuralNetwork.${exten}.${H}.TestReTest${i}.Nodes.${nodes}.out
    	# Check if model already exists
		if [ ! -f ${outFile}.h5 ]; then
    		echo "Model does not exist yet."
			nohup ${PYTHON} ${script} -dDir ${dataDir} -f ${feats} -sl ${trainingList} -hm ${hemisphere} -o ${outFile} -ds ${downSample} -l ${layers} -n ${nodes} -e ${epochs} -b ${batchSize} -r ${rate} -v 0 >& ${logFile} 2>&1&
		fi
done
