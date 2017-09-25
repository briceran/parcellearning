#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:46:33 2017

@author: kristianeschenburg
"""

"""
Methods related to downsampling.
"""

import numpy as np
import os

def subselectDictionary(keys,dictList):
    
    """
    Given a list of dictionaries, generates new down-sampled dictionaries
    using the provided list of keys.
    
    Parameters:
    - - - - -
        keys : keys to keep
        dictList : input list of dictionaries to downsample
    """
    
    outputList = []
    
    for tempDict in dictList:
        outputList.append({key: tempDict[key] for key in keys})
        
    return outputList

"""
Methods related to merging data.
"""

def mergeFeatures(dataDict,keyList):
    
    """
    Given a dictionary of featureName : dataArray, for each f in keyList, 
    merge the dataArrays into a single array.
    
    Parameters:
    - - - - - 
        yData : SubjectFeatures object "data" attribute    
        feats : features the be merged together
    """

    data = []

    for k in keyList:
        if k in dataDict.keys():
            data.append(dataDict[k])

    data = np.column_stack(data)
    
    return data

def mergeValueArrays(inDict):
    
    """
    Method to aggregate the values of a dictionary, where the values are assumed
    to be 2-D numpy arrays.
    """
    
    data = [inDict[k] for k in inDict.keys()]
    data = np.squeeze(np.row_stack(data))
    
    return data

def mergeValueLists(inDict):
    
    """
    Method to aggregate values of dictionary, where values are assumed to be
    1D lists.
    """
    
    data = [inDict[k] for k in inDict.keys()]
    data = np.squeeze(np.concatenate(data))
    
    return data

def splitArrayByResponse(data,response,responseList):
   
    """
    Method to parition feature data for all labels, from all training subjects
    in the training object -- partitions the training data features by label.
 
    Parameters:
        - - - - - 
        data : feature data array
        response : response vector
        responseList : list of possible response values
 
    Returns:
    - - - - 
        responseData : dictionary mapping response values to partitioned 
                        data arrays
    """

    responseMap = {}.fromkeys(responseList)
     
    for r in responseList:
        inds = np.where(response == r)[0]
        responseMap[r] = data[inds,:]
         
    return responseMap

def buildResponseVector(responseMap):
    
    """
    Method to build the response vector for response-specific data array.
    
    Parameters:
    - - - - - 
        responseMap : mapping of response values to data arrays
    Returns:
    - - - - 
        responseVector : mapping of response values to response vector with
                            same number of samples as data array
    """
    
    responseVector = {}.fromkeys(responseMap)

    for r in responseVector.keys():
        
        tempResponse = np.repeat(r,responseMap[r].shape[0])
        tempResponse.shape += (1,)

        responseVector[r] = tempResponse
    
    return responseVector
