#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:15:05 2017

@author: kristianeschenburg
"""

import argparse
import sys
sys.path.append('..')

import parcellearning.loaded as ld

import nibabel as nb
import numpy as np
import scipy.io as sio
import pickle

import sklearn
from sklearn import metrics

def singleLayerDice(pred,truth):
    
    labels = list(np.arange(181))
    dice = np.zeros((1,len(labels)))
    
    for lab in labels:
        indsP = np.where(pred == lab)[0]
        indsT = np.where(truth == lab)[0]
        
        overlap = len(set(indsP).intersection(set(indsT)))
        
        denom = (len(indsP) + len(indsT))
        
        if denom == 0:
            dice[0,lab] = 0
        else:
            D = (2.*overlap)/(len(indsP)+len(indsT))
            dice[0,lab] = D
        
    return dice

def regionalMisclassification(pred,truth):
    
    labels = list(np.arange(181))
    misClass = np.zeros((len(labels),2))
    
    for lab in labels:
        indsT = np.where(truth == lab)[0]
        
        labsP = pred[indsT]
        labsT = truth[indsT]
        
        mu = np.mean(labsT == labsP)
        if np.isnan(mu):
            mu = 0;
        
        misClass[lab,0] = len(indsT)
        misClass[lab,1] = mu
    
    return misClass

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

parser = argparse.ArgumentParser(description='Build training objects.')
parser.add_argument('-p','--part',help='hemisphere to process.',required=True,type=int)
args = parser.parse_args()

PART = args.part
print "Processing Part {}".format(PART)

if PART == 1:

    for itr in np.arange(N):
        
        print itr
        
        itrDir = '{}Model_{}/'.format(predDir,itr)
        subjList = '{}TestingSubjects.{}.txt'.format(testDir,itr)
        
        with open(subjList,'r') as insubj:
            subjects = insubj.readlines()
        subjects = [x.strip() for x in subjects]
        
        itrExt = 'Iteration_{}'.format(itr)
        predItrDir = '{}Model_{}/'.format(predDir,itr)
        diceDir = '{}DiceMetrics/'.format(predItrDir)
        erroDir = '{}ErrorMaps/'.format(predItrDir,itr)
        
        for hemi in hemiTypes:
            hExt = hemiMaps[hemi]
            
            print hemi
        
            for mt in methodTypes:
                
                misClassDictFile = '{}MisClass.WB.{}.{}.p'.format(erroDir,hExt,mt)
                misClassDict = {k: [] for k in dataTypes}
    
                for s,subj in enumerate(subjects):
                    
                    print subj
    
                    trueMapFile = '{}{}.{}.{}'.format(lablDir,subj,hExt,lablExt)
                    trueMap = ld.loadGii(trueMapFile)
    
                    diceMatrix_Whole = np.zeros((4,4))
                    diceMatrix_Region = np.zeros((3,181))
                    
                    jTrue = metrics.jaccard_similarity_score(trueMap,trueMap)
                    dTrue = (2.*jTrue)/(1+jTrue)
                    diceMatrix_Whole[3,3] = dTrue
                    
                    inMyl = '{}MyelinDensity/{}.{}.MyelinMap.32k_fs_LR.func.gii'.format(dataDir,subj,hExt)
                    funcObject = nb.load(inMyl)
                    
                    diceWholeFile = '{}{}.{}.{}.Dice.WB.{}.mat'.format(diceDir,subj,mt,hExt,itrExt)
                    diceRegionFile = '{}{}.{}.{}.Dice.Reg.{}.mat'.format(diceDir,subj,mt,hExt,itrExt)
                    
                    
            
                    ### Jaccard Computations ###
                    for k,DT in enumerate(dataTypes):
                        
                        errorFile = '{}{}.{}.{}.Error.{}.{}.func.gii'.format(erroDir,subj,mt,hExt,DT,itrExt)
                        errorRegFile = '{}{}.{}.{}.Error.Regional.{}.{}.mat'.format(erroDir,subj,mt,hExt,DT,itrExt)
    
                        inDTMap = '{}{}.{}.{}.{}.{}.label.gii'.format(predItrDir,subj,mt,hExt,DT,itrExt)
                        dtBaseMap = ld.loadGii(inDTMap)
                        J = metrics.jaccard_similarity_score(dtBaseMap,trueMap)
                        D = (2.*J)/(1+J)
                        
                        diceMatrix_Whole[k,3] = D
                        diceMatrix_Whole[3,k] = D
                        
                        acc = np.mean(trueMap == dtBaseMap)
                        misClassDict[DT].append(acc)
    
                        ndt = []
                        for j,nDT in enumerate(dataTypes):
    
                            pairDTMap = '{}{}.{}.{}.{}.{}.label.gii'.format(predItrDir,subj,mt,hExt,nDT,itrExt)
                            ndtBaseMap = ld.loadGii(pairDTMap)
                            J2 = metrics.jaccard_similarity_score(dtBaseMap,ndtBaseMap)
                            D2 = (2.*J2)/(1+J2)
                            ndt.append(D2)
                            
                        ndt = np.asarray(ndt)
    
                        diceMatrix_Whole[k,0:len(ndt)] = ndt
                        diceMatrix_Whole[0:len(ndt),k] = ndt
                        diceMatrix_Region[k,:] = singleLayerDice(dtBaseMap,trueMap)
    
                        errorReg = regionalMisclassification(dtBaseMap,trueMap)
                        errReg = {'errReg': errorReg}
                        sio.savemat(errorRegFile,errReg)
                    
                        errorMap = trueMap != dtBaseMap
                        funcObject.darrays[0].data = errorMap.astype(np.float32)
                        nb.save(funcObject,errorFile)
                    
                    dcmw = {'wb': diceMatrix_Whole}
                    dcmr = {'reg': diceMatrix_Region}
                    
                    sio.savemat(diceWholeFile,dcmw)
                    sio.savemat(diceRegionFile,dcmr)
                    
                    with open(misClassDictFile,'w') as outFile:
                        pickle.dump(misClassDict,outFile,-1)
                    
                for k,DT in enumerate(dataTypes):
                    
                    meanMisClass = np.zeros((32492,1))
                    meanRegMisClass = np.zeros((181,2))
                    meanMethodDiceWB = np.zeros((4,4))
                    meanMethodDiceRG = np.zeros((3,181))
                    
                    outmwMC = '{}MeanMisClass.WB.{}.{}.{}.{}.func.gii'.format(erroDir,mt,hExt,DT,itrExt)
                    outmrMC = '{}MeanMisClass.Reg.{}.{}.{}.{}.mat'.format(erroDir,mt,hExt,DT,itrExt)
                    outmmDW = '{}MeanDice.WB.{}.{}.{}.{}.mat'.format(diceDir,mt,hExt,DT,itrExt)
                    outmmDR = '{}MeanDice.Reg.{}.{}.{}.{}.mat'.format(diceDir,mt,hExt,DT,itrExt)
    
                    for s,subj in enumerate(subjects):
                        
                        errorFile = '{}{}.{}.{}.Error.{}.{}.func.gii'.format(erroDir,subj,mt,hExt,DT,itrExt)
                        errorRegFile = '{}{}.{}.{}.Error.Regional.{}.{}.mat'.format(erroDir,subj,mt,hExt,DT,itrExt)
                        
                        errorMap = nb.load(errorFile)
                        meanMisClass[:,0]+=errorMap.darrays[0].data
                        
                        errorReg = sio.loadmat(errorRegFile)
                        meanRegMisClass+=errorReg['errReg']
                        
                        
                    meanMisClass/=len(subjects)
                    meanRegMisClass/=len(subjects)
                    meanMethodDiceWB/=len(subjects)
                    meanMethodDiceRG/=len(subjects)
                
                    funcObject.darrays[0].data = np.asarray(meanMisClass).astype(np.float32)
                    nb.save(funcObject,outmwMC)
                
                    mmrmc = {'muregmc': meanRegMisClass}
                    sio.savemat(outmrMC,mmrmc)
                    
                    mmdw = {'muwb': meanMethodDiceWB}
                    sio.savemat(outmmDW,mmdw)
                    
                    mmdr = {'mureg': meanMethodDiceRG}
                    sio.savemat(outmmDR,mmdr)

if PART == 2:

    # Concatenate Regional Mean Errors Across Iterations
    for mt in methodTypes:
        for dt in dataTypes:        
            for hemi in hemiTypes:
                hExt = hemiMaps[hemi]
                
                concatFile = '{}MeanMisClass.Reg.Concatenated.{}.{}.{}.mat'.format(predDir,mt,hExt,dt)
                concatList = []
                
                print concatFile
            
                for itr in np.arange(N):
                    print itr
                    
                    meanDir = '{}Model_{}/ErrorMaps/'.format(predDir,itr)
                    meanFile = '{}MeanMisClass.Reg.{}.{}.{}.Iteration_{}.mat'.format(meanDir,mt,hExt,dt,itr)
                    
                    data = sio.loadmat(meanFile)
                    data = data['muregmc']
                    concatList.append(data)
                
                concatList = np.row_stack(concatList)
                concat = {}
                concat['muregconc'] = concatList
                sio.savemat(concatFile,concat)
                
    
       # Concatenate Regional Mean Errors Across Iterations
    for mt in methodTypes:
        for dt in dataTypes:        
            for hemi in hemiTypes:
                hExt = hemiMaps[hemi]
                
                concatFile = '{}MeanDice.WB.Concatenated.{}.{}.{}.mat'.format(predDir,mt,hExt,dt)
                concatList = []
                
                print concatFile
            
                for itr in np.arange(N):
                    print itr
                    
                    meanDir = '{}Model_{}/DiceMetrics/'.format(predDir,itr)
                    meanFile = '{}MeanDice.WB.{}.{}.{}.Iteration_{}.mat'.format(meanDir,mt,hExt,dt,itr)
                    
                    data = sio.loadmat(meanFile)
                    data = data['muwb']
                    concatList.append(data)
                
                concatList = np.row_stack(concatList)
                concat = {}
                concat['muwb'] = concatList
                sio.savemat(concatFile,concat)             
