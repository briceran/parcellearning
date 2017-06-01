# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:43:58 2017

@author: kristianeschenburg
"""

import h5py
import numpy as np
import nibabel
import os
import pickle
import scipy.io as sio


def midline(vector):
    
    """
    Method to compute the midline coordinates of a label set.
    
    Not yet tested on the output of FreeSurfer for a new IBIC subject.
        
    Assumes midline vertices will have label of 0 or -1.  Could be changed to
    accept specific value.
    
    """

    coords = np.squeeze(np.where(vector == 0))
    
    return coords

def midline_restingState(inRest,outFile):
    
    """
    
    Method to compute the midline coordinates from a resting state file.
    This is generally what we will use to pre-compute the midline indices for
    each brain, since it doesn't make sense to compute a cortical ROI for a
    a region that does not have BOLD activity.
    
    Paramters:
    - - - - - 
        
        inRest : path to resting state file
        
        outFile : name of output midline indices file
    
    """
    
    rest = loadMat(inRest)
    temp = np.sum(np.abs(rest),axis=1)
    mids = np.squeeze(np.where(temp == 0))
    
    m = {}
    m['mids'] = mids
        
    sio.savemat(outFile,m)
    

def fixMatching(matching,sN,sMids,tN,tMids):
    
    """
        
    Method to adjust the coordinates and shape of a match.
    
    For example, with the HCP data, the original data is 32492 vertices.  We
    excluded the 2796 midline vertices in the surface matching step (so we 
    included only 29696 total vertices).  The indices in the match correspond
    to positions between 1 and 29696, not 1 and 32492.
    
    This method corrects for the coordiante differences, and returns a 
    matching of length sN.
    
    Expects that the matching has already been adjusted for the
    Matlab-to-Python indexing conversion (i.e. that 1 has been subtracted).
    
    The current method works for matching between surfaces with different
    numbers of surface vertices and different midline coordinates.
    
    Paramters:
    - - - - - 
        matching : output of DiffeoSpectralMatching (corr12, corr21)    
        sN : number of vertices in full source surface    
        sMids : vector containing indices of midline for source surface    
        tN : number of vertices in full target surface    
        tMids : vector containing indices of midline for target surface
                
    Returns:
    - - - - 
        adjusted : list where matching indices are converted to range of
                    surface vertices
    """
    
    # get coordinates of vertices in target surface not in the midline
    full_coords = list(set(range(0,tN))-set(tMids))
    
    # get list of coordinates of length matching
    match_coords = list(range(0,len(matching)))

    # create dictionary mapping matching numbers to non-midline coords
    convert = dict((m,f) for m,f in zip(match_coords,full_coords))
    
    # convert the matching coordinates to non-midline coordinates
    adjusted = list(convert[x] for x in list(matching))
    
    if len(adjusted) < sN:
    
        cdata = np.zeros(shape=(sN,1))
        coords = list(set(range(0,sN)) - set(sMids))
        cdata[coords,0] = adjusted
        cdata[list(sMids),0] = -1
             
        adjusted = cdata
    
    return np.squeeze(adjusted)


def fixLabelSize(mids,dL,N):
    
    """
    Given a surface file of proper size (like the FreeSurfer Myelin Map) and
    label file that excludes the midline, adjust the size of the defunct
    label file to match that of the proper file.
    """

    coords = list(set(range(0,N))-set(mids))
    
    cdata = np.zeros(shape=(N,1))
    cdata[coords,0] = dL
    
    return np.squeeze(cdata)


def loadMat(inFile,*args):
    
    """
    Method to load .mat files.  Not part of a specific class.
    """

    if os.path.isfile(inFile):
        try:
            data = sio.loadmat(inFile)
        except NotImplementedError:
            data = h5py.File(inFile)
            data = np.transpose(np.asarray(data[data.keys()[0]]))
        else:    
            for k in data.keys():
                if k.startswith('_'):
                    del data[k]       
            data = np.squeeze(np.asarray(data.get(data.keys()[0])))   
            
        return data
    
    else:
        print('Input file does not exist.')

def loadGii(inFile,darray):
    
    """
    Method to load Gifti files.  Not part of a specific class.
    """
    
    parts = str.split(inFile,'/')
    
    try:
        data = nibabel.load(inFile)
    except OSError:
        print('Warning: {} cannot be read.'.format(parts[-1]))
    else:
        # if data is instance of GiftiImage
        if isinstance(data,nibabel.gifti.gifti.GiftiImage):
            return np.squeeze(data.darrays[darray].data)
        # if data is instance of Nifti2Image
        elif isinstance(data,nibabel.nifti2.Nifti2Image):
            return np.squeeze(np.asarray(data.get_data()))


def loadPick(inFile,*args):
    
    """
    Method to load pickle file.  Not part of a specific class.
    """

    parts = str.split(inFile,'/')

    try:
        with open(inFile,"rb") as input:
            data = pickle.load(input)
    except OSError:
        print('Warning: {} cannot be read.'.format(parts[-1]))
    else:
        return data