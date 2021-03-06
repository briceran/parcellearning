#!/bin/bash

PYTHON=/project_space/parcellation/Software/anaconda2/bin/python

kind=$1
hemisphere=$2

# Define directories
dataDir=/mnt/parcellator/parcellation/parcellearning/Data/
binDir=/mnt/parcellator/parcellation/GitHub/parcellearning/bin/
script=${binDir}GaussianMixtureModel.py

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

covType='diag'
nComp=2

if [ $hemisphere = 'Left' ]; then
	H='L'
elif [ $hemisphere = 'Right' ]; then
	H='R'
fi

outFileExtension=GMM.${H}.Covariance.${covType}.NumComponents.${nComp}.${exten}.TestReTest

echo ${outFileExtension}

for i in $(seq 1 $N); do
	outFile=${outDir}${outFileExtension}.Iteration_${i}.p
	trainingList=${dataDir}TrainTestLists/TestRetest_Train${i}.txt
	logFile=${outDir}logFile.GMM.${exten}.${H}.TestReTest.${i}.out
	if [ ! -f ${outFile}.p ]; then
    	echo "Model does not exist yet."
    	${PYTHON} ${script} -dDir ${dataDir} -f ${feats} -sl ${trainingList} -hm ${hemisphere} -o ${outFile} -cov ${covType} -nc ${nComp} >& ${logFile}
	fi
done

