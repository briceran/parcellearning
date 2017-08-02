#!/bin/bash

round=$1

PYTHON=/project_space/parcellation/Software/anaconda2/bin/python
binDir=/mnt/parcellator/parcellation/GitHub/parcellearning/bin/
script=testModelsRF_SingleRound.py

${PYTHON} ${binDir}${script} -r ${round}
