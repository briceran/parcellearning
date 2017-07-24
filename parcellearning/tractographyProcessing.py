#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 20:16:18 2017

@author: kristianeschenburg
"""

import h5py
import json
import os
import nibabel as nb
import numpy as np

def tractSpaceFromText(tractSpaceLookup):
    
    """
    Convert string coordinates to list of coordinates.
    
    Parameters:
    - - - - -
    tractSpaceLookup : lookup text file produced by probtrackx2 in matrix2 mode
    """
    
    with open(tractSpaceLookup,'r') as inCoords:
        coords = inCoords.readlines()
    coords = [x.strip() for x in coords]
    
    coords = [x.split('  ') for x in coords]
    
    coords = [np.asarray(list(map(int, x))) for x in coords]
    
    return coords

def hemiSubcorticalCoordiantes(tractSpaceCoords,rois,labelLookUp):
    
    """
    Parameters:
    - - - - -
        tractSpaceCoords : list of coordiantes from tractSpaceFromText
        
        rois : input subcortical ROI volume
        
        labelLookUp : lookup_tractspace_fdt_matrix2 file produced by 
                        probtrackX2

    Using the tract space coordinates from a given subject, 
    """
    
    label = nb.load(rois)
    labelData = label.get_data()
    
    lookup = nb.load(labelLookUp)
    lookupInts = lookup.get_data()

    mappings = {}
    
    for coord in tractSpaceCoords:
        
        coord = coord
        x = coord[0]
        y = coord[1]
        z = coord[2]
        
        # get mapped vertex integer
        mappedKey = int(lookupInts[x,y,z])
        # get label value
        data = str(int(labelData[x,y,z]))
        if data not in mappings:
            mappings[data] = list([mappedKey])
        else:
            mappings[data].append(mappedKey)
    
    return mappings


if __name__=='__main__':
    
    dataDir = '/mnt/parcellator/parcellation/HCP/Connectome_4/'
    subjectList = dataDir + 'SubjectList.txt'
    ptxExten = '/ProbTrackX2/'
    hemis = ['Left','Right']
    
    lookUpFile = 'lookup_tractspace_fdt_matrix2.nii.gz'
    coords = 'tract_space_coords_for_fdt_matrix2'
    rois = '.ROIS.acpc_dc.1.25.nii.gz'
    
    with open(subjectList,'r') as inSubjects:
        subjects = inSubjects.readlines()
    subjects = [x.strip() for x in subjects]
    
    for s in subjects:
        print 'Subject: {}'.format(s)
        subjDir = dataDir + s
        ptxDir = subjDir + ptxExten
        roiDir = subjDir + '/Structural/'
        
        for h in hemis:
            print 'Hemisphere: {}'.format(h)
            subjDirHemi = subjDir + h + '/'
        
            inLookUp = ptxDir + h + '/' + lookUpFile
            inCoords = ptxDir + h + '/' + coords
            inROIs = roiDir + h + rois
            
            outJson = ptxDir + h + '.VoxelMappings.json'
            outH5 = ptxDir + h + '.VoxelMappings.h5'
            
            cond = True
            
            if not os.path.isfile(inLookUp):
                print inLookUp
                cond = False
            if not os.path.isfile(inCoords):
                print inCoords
                cond = False
            if not os.path.isfile(inROIs):
                print inROIs
                cond = False
            
            if cond:
                
                tractCoords = tractSpaceFromText(inCoords)
                mapping = hemiSubcorticalCoordiantes(tractCoords,inROIs,inLookUp)
                
                with open(outJson,'w') as output:
                    json.dump(mapping,output)
                    
            out = h5py.File(outH5,mode='a')
            out.create_group('labels',data=mapping.keys())
            for m in mapping.keys():
                out.create_dataset(str(m),data=mapping[m])
            
            out.close()
                