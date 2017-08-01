#!/bin/bash

PYTHON=/project_space/parcellation/Software/anaconda2/bin/python

kind=$1
hemisphere=$2

# Define directories
dataDir=/mnt/parcellator/parcellation/parcellearning/Data/
binDir=/mnt/parcellator/parcellation/GitHub/parcellearning/bin/
script=${binDir}multiAtlasLabelPropagation.py

nHood=${dataDir}LabelAdjacency/HCP/Compiled.L.HCP.LabelAdjacency.p
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

depth=5
nEst=50
atlasSize=1

if [ $hemisphere = 'Left' ]; then
	H='L'
elif [ $hemisphere = 'Right' ]; then
	H='R'
fi

nHood=${dataDir}LabelAdjacency/Compiled.${H}.HCP.LabelAdjacency.p

outFileExtension=RandomForest.${Hemi}.AtlasSize.${atlasSize}.NumAtlases.Max.Depth.${depth}.NumEst.${nEst}.${exten}

echo ${outFileExtension}

for i in $(seq 0 $N); do
	outFile=${outDir}${outFileExtension}.Iteration_${i}.p
	trainingList=${dataDir}TrainTestLists/TrainingSubjects.${i}.txt
	logFile=${outDir}logFile.RandomForest.${exten}.${H}.${i}.out
	if [ ! -f ${outFile}.p ]; then
		nohup ${PYTHON} ${script} -dDir ${dataDir} -f ${feats} -sl ${trainingList} -hm ${hemisphere} -o ${outFile} -d ${depth} -nEst ${nEst} -a ${atlasSize} -nm ${nHood} >& ${logFile} 2>&1&
	fi
done

