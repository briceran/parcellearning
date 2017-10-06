#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:57:32 2017

@author: kristianeschenburg
"""

"""
Test suite defines a set of methods to assess the performance of a classifier.
These methods operate on a per-label basis to generate an H5 file for each
predicted label.  The metrics can be accessed in a similar manner to 
accessing data in a dictionary -- that is, with key-value pairs.
"""

import loaded as ld
import classifierUtilities as cu
import dataUtilities as du

import pandas as pd
import numpy as np
import nibabel as nb
from sklearn import metrics
import copy

def atlasOverlap(atlasMap,cbpLabel,A,L):
    
    """
    Compute the overlap of the predicted label file with the label file of a
    given atlas.  For example, we might want to compute the overlap of the
    connectivity-based map with the Destrieux atlas, or Desikan-Killiany atlas.
    
    Parameters:
    - - - - -
        atlasMap : dictionary, where the key is the name of the atlas map 
                    i.e. 'Destrieux', and the value is the file path
        
        cbpLabel : file path to the predicted label
        
        A : number of expected regions in atlas map
        
        L : number of expected regions in the predicted map
    """
    
    atlName = atlasMap['name']
    atlFile = atlasMap['file']
    
    atl = nb.load(atlFile)
    atl = atl.darrays[0].data
    atlLabels = list(set(atl).difference({0}))
    print atlLabels
    
    cbp = nb.load(cbpLabel)
    cbp = cbp.darrays[0].data
    cbpLabels = list(set(cbp).difference({0}))
    
    overlaps = np.zeros((L+1,A+1))
    
    cbpIndices = {}.fromkeys(np.arange(1,L))
    atlIndices = {}.fromkeys(np.arange(1,A+1))
    
    for c in cbpLabels:
        cbpIndices[c] = np.where(cbp == c)[0]
    
    for a in atlLabels:
        atlIndices[a] = np.where(atl == a)[0]
    
    print 'Entering loop'
    for c in cbpLabels:
        cbpInds = cbpIndices[c]
        
        for a in atlLabels:
            atlInds = atlIndices[a]
            
            if len(atlInds) and len(cbpInds):
                
                ov = len(set(cbpIndices[c]).intersection(set(atlIndices[a])))
                overlaps[c,a] = (1.*ov)/len(cbpIndices[c])
            else:
                overlaps[c,a] = 0
    
    return [atlName,overlaps]

def accuracy(label1,label2):
    
    """
    Compute the classification accuracy of the predicted label as compared to
    the ground truth label.
    """

    return np.mean(label1 == label2)

def jaccard(label1,label2):
    
    """
    Compute the overlap of 2 predictions using Jaccard Index.  Returns a mean score
    """
    
    return metrics.jaccard_similarity_score(label1,label2)

def errorMap(label1,label2):
    
    """
    Generate an error map of where the label1 == label2.
    """
    
    return 1.*(np.asarray(label1) == np.asarray(label2))

if __name__ == "__main__":
   
    samples = 'Core'
 
    baseDirectory = '/mnt/parcellator/parcellation/parcellearning/Data/'
    
    predictionDirectory = '{}Predictions/TestReTest/NeuralNetwork/Deep/'.format(baseDirectory)
    truthDirectory = '{}Labels/HCP/'.format(baseDirectory)

    testList= '{}TrainTestLists/TestRetest_Test.txt'.format(baseDirectory)
    testSubjects = cu.loadList(testList)

    results = []

    hemi = ['L','R']
    dataTypes = ['Full','RestingState','ProbTrackX2']

    for d in dataTypes:
        
        print 'DataType: {}'.format(d)

        dataTypeDirectory = '{}{}/'.format(predictionDirectory,d)
        
        for test_subj in testSubjects:
            
            print 'Subject: {}'.format(test_subj)
            
            for h in hemi:

                params = [test_subj,d,h]
                
                print 'Hemisphere: {}'.format(h)
                
                truth = '{}{}.{}.CorticalAreas.fixed.32k_fs_LR.label.gii'.format(truthDirectory,
                                     test_subj,h)
                truth = nb.load(truth)
                truth = truth.darrays[0].data
                
                print 'True label loaded.'

                test1 = '{}{}.{}.{}.Deep2.TestReTest_1.func.gii'.format(dataTypeDirectory,testSubj,h,d)
                test1 = nb.load(test1)
                test1 = test1.darrays[0].data

                test2 = '{}{}.{}.{}.Deep2.TestReTest_2.func.gii'.format(dataTypeDirectory,testSubj,h,d)
                test2 = nb.load(test2)
                test2 = test2.darrays[0].data

                error1 = errorMap(truth,test1)
                error2 = errorMap(truth,test2)
                errort = errorMap(test1,test2)

                acc1 = np.mean(error1)
                acc2 = np.mean(error2)
                acct = np.mean(errort)

                params.append(acc1)
                params.append(acc2)
                params.append(accT)

                results.append(params)

    results = np.row_stack(results)
    dataFrame = pd.DataFrame(results)
    
    pNames = ['id','data','hemi','acc_1','acc_2','acc_test']
    dataFrame.columns = pNames
    
    dataFrame.to_pickle('/mnt/parcellator/parcellation/parcellearning/Data/TestReTestData.Deep.Core.p'.format(samples))
                            
                            
                            
                            
                            
                            
                            
                            
                        
                        
    
    
