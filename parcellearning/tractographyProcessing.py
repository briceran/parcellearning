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

def hemiCorticalCoordiantes(inLabel):
    
    """
    Parameters:
    - - - - -
        inLabel : input label file (Desikan-Killiany, Destrieux, HCP) with
                    spatial coordiantes corresponding to surface file used
                    to seed probtrackx2
    """
    
    label = nb.load(inLabel)
    label = label.darrays[0].data
    
    values = set(label).difference({0})
    
    mappings = {}.fromkeys(values)
    
    for m in mappings:
        inds = np.where(label == m)[0] + 1
        mappings[str(m)] = list(inds)
    
    return mappings


if __name__=='__main__':
    
    dataDir = '/mnt/parcellator/parcellation/HCP/Connectome_4/'
    subjectList = dataDir + 'SubjectList.txt'
    ptxExten = '/ProbTrackX2/'
    hemis = ['Left','Right']
    
    lookUpFile = 'lookup_tractspace_fdt_matrix2.nii.gz'
    coords = 'tract_space_coords_for_fdt_matrix2'
    roisExt = '.ROIS.acpc_dc.1.25.nii.gz'
    
    rois = ['Accumbens','Amygdala','Caudate','Hippocampus',
            'Pallidum','Putamen','Thalamus']
    
    lValues = [26,18,11,17,13,12,10]
    rValues = [58,54,53,50,52,51,49]
    values = [zip(rois,lValues),zip(rois,rValues)]
    
    roi_values = {}.fromkeys(hemis)
    roi_values['Left'] = dict(values[0])
    roi_values['Right'] = dict(values[1])
    
    with open(subjectList,'r') as inSubjects:
        subjects = inSubjects.readlines()
    subjects = [x.strip() for x in subjects]
    
    ## Process sub-cortical tractography data
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
            inROIs = roiDir + h + roisExt
            
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
                    
            roiMaps = roi_values[h]
            r = roiMaps.keys()
            l = roiMaps.values()
            
            if not os.path.isfile(outH5):
                    
                out = h5py.File(outH5,mode='a')
                out.attrs.create('regions',rois)
                out.create_group('regionValues')
                for r in rois:
                    out['regionValues'].attrs.create(r,roi_values[h][r])
                    
                for m in mapping.keys():
                    out.create_dataset(str(m),data=np.asarray(mapping[m]))
                
                out.close()
    
    ## Process Cortical tractography data
    hemiExten = ['L','R']
    labelDir = '/mnt/parcellator/parcellation/parcellearning/Data/Labels/'
    
    labs = ['Desikan','Destrieux','HCP']
    labExts = ['aparc','aparc.a2009s','CorticalAreas.fixed']
    
    for s in subjects:
        
        subjDir = dataDir + s
        ptxDir = subjDir + ptxExten
        
        for index,LAB in enumerate(zip(labs,labExts)):
            for k,HEMI in enumerate(hemiExten):
                
                h = '.' + hemiExten[k] + '.'
                lExt = h + LAB[1] + '.32k_fs_LR.label.gii'
                labelFile = labelDir + LAB[0] + '/' + s + lExt
                print labelFile
                
                if os.path.isfile(labelFile):
                    mapping = hemiCorticalCoordiantes(labelFile)
                    
                    outJson = ptxDir + hemis[k] + '.VertexMappings.json'
                    outH5 = ptxDir + hemis[k] + '.VertexMappings.h5'
                    
                    with open(outJson,'w') as output:
                        json.dump(mapping,output)
                        
                    if not os.path.isfile(outH5):
                    
                        out = h5py.File(outH5,mode='a')
                        out.attrs.create('regions',mapping.keys())
                        
                        for m in mapping.keys():
                            out.create_dataset(str(m),data=np.array(mapping[m]))
                        
                        out.close()
    
    
                