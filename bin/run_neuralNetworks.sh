#!/bin/bash

PYTHON=/project_space/parcellation/Software/anaconda2/bin/python

kind=$1
hemisphere=$2
layers=$3
nodes=$4
outDir=$5

dataDir=/mnt/parcellator/parcellation/parcellearning/Data/
binDir=/mnt/parcellator/parcellation/GitHub/parcellearning/bin/
script=${binDir}neuralNetworks.py

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

# If number of layers is undefined, default = 3
if [ -z "$3" ]; then
	layers=3
fi

# If number of nodes is undefined, default = 150
if [ -z "$4" ]; then
	nodes=150
fi

# If output directory is undefined, default is generic "models" directory
if [ -z "$5" ]; then
	outDir=${dataDir}Models/
fi
# and if output directory does not exist, created
if [ ! -d "$outDir" ]; then
	mkdir ${outDir}
fi

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

if [ $hemisphere = 'Left' ]; then
	H='L'
elif [ $hemisphere = 'Right' ]; then
	H='R'
fi

#layers=2
#nodes=150

downSample='equal'
epochs=40
batchSize=256
rate=0.001

for i in $(seq 0 $N); do
		outFileExtension=NeuralNetwork.${H}.Layers.${layers}.Nodes.${nodes}.Sampling.${downSample}.Epochs.${epochs}.Batch.${batchSize}.Rate.${rate}.${exten}
		echo ${outFileExtension}
		outFile=${outDir}${outFileExtension}.Iteration_${i}
		trainingList=${dataDir}TrainTestLists/TrainingSubjects.${i}.txt
		logFile=${outDir}logFile.NeuralNetwork.${exten}.${H}.${i}.Nodes.${nodes}.out
    	# Check if model already exists
		if [ ! -f ${outFile}.h5 ]; then
    		echo "Model does not exist yet."
			nohup ${PYTHON} ${script} -dDir ${dataDir} -f ${feats} -sl ${trainingList} -hm ${hemisphere} -o ${outFile} -ds ${downSample} -l ${layers} -n ${nodes} -e ${epochs} -b ${batchSize} -r ${rate} >& ${logFile} 2>&1&
		fi
done
