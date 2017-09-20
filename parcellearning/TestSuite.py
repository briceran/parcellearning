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

import numpy as np
import nibabel as nb
from sklearn import metrics
import copy

def loadData(yObject,features):
        
        """
        Method to load the test data into the object.  We might be interested
        in loading new test data into, so we have explicitly defined this is
        as a method.
        
        Parameters:
        - - - - -
            y : SubjectFeatures object for a test brain   
            
            yMatch : MatchingFeaturesTest object containing vertLib attribute 
                    detailing which labels each vertex in surface y maps to 
                    in the training data
                    
            features : feature to included in the test data numpy array
                        these must be the same as the training data
        """

        nf = []
        for f in features:
            if f != 'label':
                nf.append(f)

        # load test subject data, save as attribtues
        tObject = ld.loadH5(yObject,*['full'])
        ID = tObject.attrs['ID']

        parsedData = ld.parseH5(tObject,nf)
        tObject.close()

        data = parsedData[ID]
        x_test = cu.mergeFeatures(data,nf)

        return x_test

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

def modelHomogeneity(data,truthLabel,cbpLabel,L,iters):
    
    """
    Compute homogeneity for a predicted label, its truth label, and random
    permutations of the predicted label.
    
    Parameters:
    - - - - -
        truthLabel : true cortical map
        
        cbpLabel : predicted cortical map
        
        L : expected number of regions
        
        iters : number of random permutations
    """
    
    truth = copy.deepcopy(truthLabel)
    cbp = copy.deepcopy(cbpLabel)

    hmg = {}.fromkeys(['truth','predicted','random'])
    hmg['random'] = {}.fromkeys(np.arange(1,L+1))

    hmg['truth'] = homogeneity(truth,data,180)
    hmg['predicted'] = homogeneity(cbpLabel,data,180)
    
    permutes = np.zeros((L+1,iters))
    
    for k in np.arange(0,iters):
        
        cbp = copy.deepcopy(cbp)

        np.random.shuffle(cbp)
        p = homogeneity(cbp,data,L)
        permutes[p.keys(),k] = p.values()
        print permutes[:,1] == permutes[:,2]

    for lab in np.arange(1,L+1):
        hmg['random'][lab] = permutes[lab,:]
    
    return (hmg,permutes)

def homogeneity(labelArray,trainingData,L):
    
    """
    Mean the homogeneity of each region with regard to its feature vectors.
    
    Parameters:
    - - - - -
        labelArray : vector of labels
        
        L : expected number of labels
    """
    
    regional = {}.fromkeys(np.arange(1,L+1))
    
    for lab in regional.keys():
        inds = np.where(labelArray == lab)[0]
        if len(inds) > 1:
            data = trainingData[inds,:]
            sims = np.corrcoef(data)
            regional[lab] = np.mean(sims)
        elif len(inds) == 1:
            regional[lab] = 1
        else:
            regional[lab] = 0;
            
    return regional

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

def silhouette(data,labels):
    
    """
    Compute the silhouette score for a whole parcellation, and the individual 
    scores for each sample.
    """
    
    score = metrics.silhouette_score(data,labels,metric='correlation')
    sample_scores = metrics.silhouette_samples(data,labels,metric='correlation')
    
    return [score,sample_scores]

def testSuite(truth,predicted,featureData):
    
    """
    Generate a suite of metrics associated with a predicted labeling.
    """

    acc = accuracy(predicted,truth)
    [shScore,shSamples] = silhouette(featureData,predicted)
    
    return [acc,shScore,shSamples]


if __name__ == "__main__":
    
    baseDir = '/mnt/parcellator/parcellation/parcellearning/Data/'
    testFile = '{}TrainTestLists/TestRetest_Test.txt'.format(baseDir)
    
    predDir =  '{}Predictions/TestReTest/NeuralNetwork/'.format(baseDir)
    trueDir = '{}Labels/HCP/'.format(baseDir)
    
    dataDir = '{}TrainingObjects/FreeSurfer/'.format(baseDir)
    
    rate = 0.001
    samples = 'equal'
    epochs = 40
    batch = 256
    
    midExt = 'Sampling.{}.Epochs.{}.Batch.{}.Rate.{}'.format(samples,epochs,
                       batch,rate)
    
    tRt = [1,2]
    freqs = [0,1,2]
    layers = [2,3]
    nodes = [25,50]
    hemi = ['L','R']
    
    dataTypes = ['Full','ProbTrackX2','RestingState']
    dataFeatures = ['fs_cort,fs_subcort,sulcal,myelin,curv',
                'pt_cort,pt_subcort,sulcal,myelin,curv',
                'fs_cort,fs_subcort,pt_cort,pt_subcort,sulcal,myelin,curv']
    dataFeatureFunc = dict(zip(dataTypes,dataFeatures))

    with open(testFile,'r') as inTest:
        testSubjects = inTest.readlines()
    testSubjects = [x.strip() for x in testSubjects]
    
    results = []
    
    for d in dataTypes:
        
        print 'DataType: {}'.format(d)
        features = dataFeatureFunc[d]
        features = features.split(',')
        
        predDataDir = '{}{}/'.format(predDir,d)
        
        for test_subj in testSubjects:
            
            print 'Subject: {}'.format(test_subj)
            
            for h in hemi:
                
                print 'Hemisphere: {}'.format(h)
                
                truth = '{}{}.{}.CorticalAreas.fixed.32k_fs_LR.label.gii'.format(trueDir,
                                     test_subj,h)
                truth = nb.load(truth)
                truth = truth.darrays[0].data
                
                print 'True label loaded.'
                
                mylFile = '{}MyelinDensity/{}.{}.MyelinMap.32k_fs_LR.func.gii'.format(baseDir,
                       test_subj,h)
                mylData = nb.load(mylFile)
                myl = mylData.darrays[0].data
                
                print 'Functional map loaded.'
                
                subjData = '{}{}.{}.TrainingObject.aparc.a2009s.h5'.format(dataDir,test_subj,h)
                x_test = loadData(subjData,features)
                
                print 'Feature data loaded.'
                
                for f in freqs:
                    f = float(f)
                    
                    print 'Frequency Power: {}'.format(f)
                    
                    for n in nodes:
                        
                        print 'Nodes: {}'.format(n)
                        for l in layers:
                            
                            print 'Layers: {}'.format(l)
                            
                            params = [test_subj,d,h,f,n,l]
                            
                            midPre = '{}.Layers.{}.Nodes.{}'.format(h,l,n)
                            midSuf = 'Freq.{}.{}.TestReTest_'.format(f,d)
                            
                            fullExt = '{}.{}.{}.{}'.format(test_subj,midPre,midExt,midSuf)
                            
                            test1 = '{}{}1.func.gii'.format(predDataDir,fullExt)
                            test1 = nb.load(test1)
                            test1 = test1.darrays[0].data
                            
                            test2 = '{}{}2.func.gii'.format(predDataDir,fullExt)
                            test2 = nb.load(test2)
                            test2 = test2.darrays[0].data
                            
                            [acc1,sc1,sh1] = testSuite(truth,test1,x_test)
                            [acc2,sc2,sh2] = testSuite(truth,test2,x_test)
                            
                            
                            error1 = errorMap(truth,test1)
                            error2 = errorMap(truth,test2)
                            errorT = errorMap(test1,test2)

                            jcc1 = jaccard(truth,test1)
                            jcc2 = jaccard(truth,test2)
                            jccT = jaccard(test1,test2)
                            accT = accuracy(test1,test2)
                            
                            params.append(jcc1)
                            params.append(jcc2)
                            params.append(jccT)
                            params.append(accT)
                            
                            results.append(params)
                            
                            
                            
                            
                            
                            
                            
                            
                        
                        
    
    