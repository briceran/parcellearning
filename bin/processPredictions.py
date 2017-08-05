#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:15:05 2017

@author: kristianeschenburg
"""

sys.path.append('..')
import parcellearning.loaded as ld

import nibabel as nb
import numpy as np
import scipy.io as sio
from sklearn import metrics

def singleLayerDice(pred,truth):
    
    labels = list(np.arange(181))
    dice = np.zeros((len(labels),1))
    
    for lab in labels:
        indsP = np.where(pred == lab)[0]
        indsT = np.where(truth == lab)[0]
        
        D = (2*np.sum(indsP==indsT))/(len(indsP)+len(indsT))
        dice[lab] = D
        
    return dice
        

methodTypes = ['NeuralNetwork','RandomForest','GMM']
dataTypes = ['RestingState','ProbTrackX2','Full']

hemiTypes = ['Left','Right']
hemiExts = ['L','R']
hemiMaps = dict(zip(hemiTypes,hemiExts))

dataDir = '/mnt/parcellator/parcellation/parcellearning/Data/'
predDir = '{}Predictions/'.format(dataDir)
lablDir = '{}Labels/HCP/'.format(dataDir)
lablExt = 'CorticalAreas.fixed.32k_fs_LR.label.gii'
testDir = '{}TrainTestLists/'.format(dataDir)

N = 10

for itr in np.arange(N):
    
    itrDir = '{}Model_{}/'.format(itr)
    subjList = '{}TestingSubjects.{}.txt'.format(testDir,itr)
    
    with open(subjList,'r') as insubj:
        subjects = insubj.readlines()
    subjects = [x.strip() for x in subjects]
    
    itrExt = 'Iteration_{}'.format(itr)
    predItrDir = '{}Model_{}/'.format(predDir,itr)
    diceDir = '{}/DiceMetrics'.format(predItrDir,itr)
    erroDir = '{}/ErrorMaps/'.format(predItrDir,itr)
    
    for hemi in hemiTypes:
        hExt = hemiMaps[hemi]
    
        for s,subj in enumerate(subjects):
            for mt in methodTypes:
            
                trueMapFile = '{}{}.{}.{}'.format(lablDir,subj,hExt,lablExt)
                trueMap = ld.loadGii(trueMapFile)

                diceMatrix_Whole = np.zeros((4,4))
                diceMatrix_Region = np.zeros((3,181))
                
                inMyl = '{}MyelinDensity/{}.{}.MyelinMap.32k_fs_LR.func.gii'.format(dataDir,subj,hExt)
                funcObject = nb.load(inMyl)
        
                ### Jaccard Computations ###
                for k,DT in enumerate(dataTypes):
                    
                    diceWholeFile = '{}{}.{}.{}.{}.Dice.WB.{}.mat'.format(diceDir,subj,mt,hExt,DT,itrExt)
                    diceRegionFile = '{}{}.{}.{}.{}.Dice.Reg.{}.mat'.format(diceDir,subj,mt,hExt,DT,itrExt)
                    errorFile = '{}{}.{}.{}.{}.Error.{}.mat'.format(erroDir,subj,mt,hExt,DT,itrExt)

                    inDTMap = '{}{}.{}.{}.{}.{}.label.gii'.format(predItrDir,subj,mt,hExt,DT,itrExt)
                    dtBaseMap = ld.loadGii(inDTMap)
                    J = metrics.jaccard_similarity_score(dtBaseMap,trueMap)
                    D = (2.*J)/(1+J)
                    
                    diceMatrix_Whole[k,3] = D
                    diceMatrix_Whole[3,k] = D

                    ndt = []
                    for nDT in dataTypes:
                        
                        pairDTMap = '{}{}.{}.{}.{}.{}.label.gii'.format(predItrDir,subj,mt,hExt,nDT,itrExt)
                        ndtBaseMap = ld.loadGii(pairDTMap)
                        J2 = metrics.jaccard_similarity_score(dtBaseMap,ndtBaseMap)
                        D2 = (2.*J)/(1+J)
                        ndt.append(D2)
                    
                    diceMatrix_Whole[0:2,0:2] = ndt.reshape(3,3)
                    diceMatrix_Region[k,:] = singleLayerDice(dtBaseMap,trueMap)
                    
                    dcmw = {'wb': diceMatrix_Whole}
                    dcmr = {'reg': diceMatrix_Region}
                    
                    sio.savemat(diceWholeFile,dcmw)
                    sio.savemat(diceRegionFile,dcmr)
                    
                    errorMap = trueMap == dtBaseMap
                    funcObject.darrays[0].data = errorMap.astype(np.float32)
                    nb.save(funcObject,errorFile)
                    