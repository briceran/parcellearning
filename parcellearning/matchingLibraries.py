# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:29:43 2017

@author: kristianeschenburg
"""

import loaded as ld
import numpy as np
import pickle

import copy
import os

import time

"""

We compute two types of matching libraries: training and testing.

We have a set of training data (i.e. HCP brains, with their own parcellations)
We perform DiffeoSurfaceMatching between all of the HCP brains.  For each
training brain, we determine the set of labels that are consistenly mapped
to each vertex and each label: vertexLibrary and labelLibrary.

In the training data, these summarize the labels that a given vertex or label
is confused with.

In the testing data, we do not have a set of connectivity-derived labels -- 
we only have the FreeSurfer labels.  We perform DiffeoSurfaceMatching between
a test brain and all training brains.  For each vertex in test brain, we 
summarize the labels from the train data that the test vertex is mapped to.

We build mappedLibrary (contains the matched labels), labelLibrary appet(confusion
label data from the training labelLibrary), and vertexLibrary (confusion
label data from the training vertexLibrary).

We expect that the MatchingLibraryTrain has been run first, for all of the 
relevent training brains -- this is input into MatchingLibraryTest.

"""

class MatchingLibraryTrain(object):
    
    """
    Computes the label-resolution and vertex-resolution matching libraries for
    a given subject and parcellation scheme.
    
    A library is a dictionary structure, where each key is a vertex ID or
    label ID.  For both library types, the values of each key are the set of 
    labels that are consistently mapped to it.
    
    The Label library will have inherently larger sets than the vertex
    library because it aggregates data.  Likewise, both libraries will grow 
    as more matches are added.
    
    Parameters:
    - - - - -

        target : target subject ID
        
        target_label : target brain label file
        
        target_myl : surface map with proper midline structure
        
        target_surface : target brain surface gifti file
        
    """
    
    def __init__(self,target,target_label,target_mids,target_surface):
        
        # Load input data, and initialie data attribtues
        self.ID = target
        
        # we load the surface as a source of control
        # we need to determine if our label file is the same size as the 
        # surface or if it needs to be adjusted
        verts = ld.loadGii(target_surface,0)
        
        # get correct number of vertices
        self.N = verts.shape[0]
        self.label = ld.loadGii(target_label,0)
        
        self.mids = ld.loadMat(target_mids)-1
        
        if self.N != len(self.label):
            
            print('Warning:  Target surface has more vertices that '\
                            'label file.  Adjusting label file.')
            
            self.label = ld.fixLabelSize(self.mids,self.label,self.N)
            
            
        # Initialize library attribtues
        # vertCounts contains the labels a vertex-of-interest maps to
        self.vertCounts = {}
        self.vertCounts = self.vertCounts.fromkeys(list(np.arange(self.N)))

        # labCounts contains the labels a label-of-interest maps to
        self.labCounts = {}
        self.labCounts = self.labCounts.fromkeys(list(self.label))
        
        self.matchedSubjects = set([])
        
    def buildLibraries(self,source,source_label,source_mids,source_surf,matching):
        
        """
        Updates the vertex and label libraries with matching results.
        
        Parameters:
        - - - - -
            
            source : source subject ID
            
            source_label : source label file
            
            matching : source-to-target matching file produed by 
                        DiffeoSpectralMatching (corr12 or corr21)
        """
        
        # vert vertices of source surface
        verts = ld.loadGii(source_surf,0)
        
        # get number of vertices in source surface
        sN = verts.shape[0]
        
        # load source label and midline vertices
        sLab = ld.loadGii(source_label,0)
        sMids = ld.loadMat(source_mids)-1
        
        # check to make sure label file has same number of vertices as 
        # surface
        if len(sLab) != sN:
            
            sLab = ld.fixLabelSize(sMids,sLab,sN)
        
        # load source-to-target matching 
        s2t = np.squeeze(ld.loadMat(matching) - 1).astype(int)

        # fix matching
        # target vertices in target space, source vertices correct length
        # s2t is of length (sN-mids), with indices in that range
        # will become length sN with indices in correct range
        fixed = ld.fixMatching(s2t,sN,sMids,self.N,self.mids).astype(int)

        # check if source subject already added
        if source not in self.matchedSubjects:
            
            # check to make sure matching / source label file same length
            if len(fixed) == len(sLab):
            
                # add matched subject to list of seen
                self.matchedSubjects.add(source)
                
                for node in np.arange(len(fixed)):
                
                    # get target vertex to which source vertex is mapped
                    vertex = fixed[node]
                    # get label of source vertex
                    sl = sLab[node]
                    
                    # if target vertex and source label not midline
                    if vertex != -1 and sl > 0:
                        
                        # get target vertex label
                        tl = self.label[vertex]
        
                        # check if current vertex already in vertexLibrary.keys
                        if not self.vertCounts[vertex]:
                            self.vertCounts[vertex] = {sl:1}
                        else:
                            if sl not in self.vertCounts[vertex].keys():
                                self.vertCounts[vertex].update({sl: 1})
                            else:
                                self.vertCounts[vertex][sl] += 1

                        if not self.labCounts[tl]:
                            self.labCounts[tl] = {sl: 1}
                        else:
                            if sl not in self.labCounts[tl].keys():
                                self.labCounts[tl].update({sl: 1})
                            else:
                                self.labCounts[tl][sl] += 1
            else:
                print('Warning:  Matching not the same length as '\
                                'source label file.')
        else:
            print('Source has already been included in the libraries.')

    def saveLibraries(self,obj,*args):
        
        """
        Save whole MatchingLibrary object, or individual libraries.
        
        Parameter:
        - - - - -
            
            obj : boolean indicating whether you want to save the whole object,
            or just the libraries.
            
            *args : file names, depending on what you want to save
        """
                
        if obj:
            try:
                with open(args[0],"wb") as output:
                    pickle.dump(self,output,-1)
            except IndexError:
                print('If obj == True, must provide filename to save object.')
        else:
            try:
                with open(args[0],"wb") as output:
                    pickle.dump(self.vertCounts,output,-1)
                with open(args[1],"wb") as output:
                    pickle.dump(self.labCounts,output,-1)
            except IndexError:
                print('If obj == False, must provide two file names to save '\
                      'vertexLibrary and labelLibrary.')


class MatchingLibraryTest(object):
    
    """
    Computes the matching libraries for a test subject.
    
    This is different from the matching libraries for the training
    subjects:  for a test subject, we are interested in which training labels 
    a test vertex maps to.  This will help us determine which classifers to
    consider.
    
    Parameters:
    - - - - -
        
        source : test subject ID
        
        source_label : test subject label set (FreeSurfer)
        
        source_surface : test subject surface file
    
    """
    
    def __init__(self,source,source_label,source_mids,source_surface):
        
        self.ID = source
        
        verts = ld.loadGii(source_surface,0)
        
        self.N = verts.shape[0]
        self.label = ld.loadGii(source_label,0)
        
        self.mids = ld.loadMat(source_mids)-1
        
        if self.N != len(self.label):
            
            print('Warning:  Surface has more vertices that '\
                            'label file.  Adjusting label file.')
            
            self.label = ld.fixLabelSize(self.mids,self.label,self.N)
        
        self.vertLib = {}
        
    def addToLibraries(self,train,trainML,match):
        
        """
        Updates the testing subject libraries with the results of a specific
        matching.
        
        Parameters:
        - - - - -
            train : train subject ID
            trainML : train subject MatchingLibraryTrain
            match : test-to-train matching file
        """
        
        if not self.vertLib:
            
            r = np.arange(0,self.N)
            
            # VertexLibrary
            # Contains the unique labels that a given vertex maps to, and the 
            # number of times the vertex maps to this label
            
            self.vertLib = {}
            self.vertLib = self.vertLib.fromkeys(list(r))
            
        # load SubjectFeatures training data object
        train = ld.loadPick(trainML)
        print train.__dict__.keys()
        
        # load test to train matching
        match = np.asarray(np.squeeze(ld.loadMat(match) - 1).astype(int))
        
        gCoords = np.asarray(list(set(np.arange(train.N)).difference(set(train.mids))))
        cCoords = np.asarray(list(set(np.arange(self.N)).difference(set(self.mids))))

        fixed = np.squeeze(np.zeros((self.N,1)))
        fixed[cCoords] = gCoords[match]
        fixed = fixed.astype(np.int32)
        
        

        # fixed matching
        # fixed = ld.fixMatching(match,self.N,self.mids,train.N,train.mids)
        # fixed = fixed.astype(int)
        
        # make sure matching is same length as source label
        if len(fixed) == self.N:
            # for each vertex in source brain
            for node in np.arange(0,len(fixed)):
                # get matched vertex in target brain
                vertex = fixed[node]
                
                # make sure target coordinate is not in midline
                if vertex != -1 and self.label[node] > 0:
                    # get target label
                    tL = train.label[vertex]
                    
                    # update labelLibrary
                    if not self.vertLib[node]:
                        self.vertLib[node] = {tL: 1}
                    else:
                        if tL not in self.vertLib[node].keys():
                            self.vertLib[node][tL] = 1
                        else:
                            self.vertLib[node][tL] += 1
                            
    def uniqueMappings(self):
    
        """
        Method to decompose self.vertLib into all of the unique label mappings 
        of the test brain vertices.  Should be run after all training subject
        mappings have been added.
        
        mapID maps an identifying value to each unique mapped label set
        
        mapVerts maps the same identifying values to the set of vertices with
        that label set
            
        """
        mapSets = {}
        
        
        for k in self.vertLib.keys():
            maps = self.vertLib[k]
            
            if maps:    
                mapSets[k] = set(maps.keys())
                
        # for each value in mapSets (a set of labels)
        # convert value to tuple
        # add tuple to a set -- we now have all unique tuples without duplicates
        # then convert set to non-duplicates list of tuples
        start = time.time()
        unique_maps = [list(x) for x in set(tuple(x) for x in mapSets.values())]
        end = time.time()
        print('Time to find unique sets of labels: ',(end-start),' sec')
        
        mapID = {}
        mapVerts = {}
        
        # iterate over unique label set tuples in unique_maps
        start = time.time()
        for en,mp in enumerate(unique_maps):

            # return the vertices that have the unique label set mappings
            verts = [k for k,v in mapSets.iteritems() if list(v) == mp]
            # assign unique label set to ID 'en'
            mapID[en] = mp
            # assign unique list of verties to ID 'en'
            mapVerts[en] = list(verts)
            
        end = time.time()
        print('Time to map vertex sets to label sets: ',(end-start),' sec')
        
        self.mapID = mapID
        self.mapVerts = mapVerts
        
    def saveLibraries(self,outFile):
        
        """
        Method to save .vertLib attribute.
        """
        
        if self.vertLib:
            try:
                with open(outFile,"wb") as outLib:
                    pickle.dump(self,outLib,-1)
            except IOError:
                print('Cannot write {} testing library to file.'.format(self.ID))
        else:
            print('Cannot write empty vertLib attribute.')
            
def mergeMappings(subjects,inputDir,exten,normalize=False):
    
    """
    Method to merge MatchingLibraryTrain objects.  For a given object, we have
    vertCounts and labCounts -- here we merge the labCounts attributes to
    return a dictionary that has the aggregate maps.
    
    The keys are labels, and the values are lists of labels that each key label
    maps to.
    
    Parameters:
    - - - - -
        subjects : list of training subjects to include in the merged object    
        inputDir : input directory where the MatchingLibraryTrain objects exist    
        exten : exten of the MatchingLibraryTrain objects
        
    Returns:
    - - - -
        merged : dictionary containing the aggregated labCounts_ results for
                    each included MatchingLibraryTrain object.  We do not
                    keep track of the counts.
    """
    
    cond = True
    merged = {}
    
    if not os.path.isdir(inputDir):
        cond = False
        print('Input directory does not exist.')
        
    if cond:
        for s in subjects:
            inMTL = inputDir + s + exten
            
            if not os.path.isfile(inMTL):
                print('MatchingLibrary for {} does not exist. Skipping.'.format(s))
            else:
                mtl = ld.loadPick(inMTL)
                merged = addSingleLibrary(merged,mtl)
    if normalize:
        merged = mappingFrequency(merged)
    
    return merged

def addSingleLibrary(merged,mtl):
    
    """
    Method to add a single library to the aggregating merged library.
    
    Parameters:
    - - - - - 
        merged : aggregating dictionary     
        mtl : loaded MatchingLibraryTrain object.  mtl contains a dictionary, 
                where keys, K, are a single label, and values are a sub-dict.
                This sub-dict has keys that are labels, L, and counts, C, 
                indicating the number of times label L, mapped to label K.
    Returns:
    - - - -
        merged : aggregating dictionary, updated with the current 
                    MatchingLibraryTrain object         
    """
    
    labs = mtl.labCounts.keys()

    for l in labs:
        # check that label, K, has mapping information
        if mtl.labCounts[l]:
            # if it does, get labels L that map to it
            maps = mtl.labCounts[l]
            
            if l not in merged.keys():
                merged[l] = maps
            else:
                for key in maps:
                    if key not in merged[l]:
                        merged[l].update({key:maps[key]})
                    else:
                        merged[l][key] += maps[key]
    return merged

def mappingFrequency(merged):
    
    """
    Convert matching library counts to frequencies.  We will use this for
    thresholding during the classification step.
    
    Parameters:
    - - - - -
        merged : merged MatchingLibrary file
    Returns:
    - - - -
        merged : merged MatchingLibrary object with normalized counts
    """
    
    mergedC = copy.deepcopy(merged)
    
    if isinstance(mergedC,str):
        mergedC = ld.loadPick(mergedC)
    elif isinstance(mergedC,dict):
        mergedC = mergedC
    
    for lab in mergedC.keys():
        if mergedC[lab]:
            maps = mergedC[lab].keys()
            total = 1.*np.sum(mergedC[lab].values())
            
            for m in maps:
                mergedC[lab][m] /= (1.*total)
    return mergedC

def mappingConfusionMatrix(merged):
    
    """
    Method to build a confusion matrix from the merged library data.  Note that
    this is not a symmetric matrix.  Rather, we simply display the mapping
    frequencies in an array, where each row coresponds to a target label, and
    each index in a row corresponds to the frequency with which a source label
    maps to the target.
    
    Parameters:
    - - - - -
        merged : merged MatchingLibrary file
        
    Returns:
    - - - -
        confusion : array of size N labels by N labels
    """
    
    if isinstance(merged,str):
        merged = ld.loadPick(merged)
    elif isinstance(merged,dict):
        merged = merged
        
    labels = merged.keys(); N = len(labels)
    mappings = dict(zip(labels,np.arange(len(labels))))
    
    confusion = np.zeros((N,N))
    
    for lab in labels:
        if merged[lab]:
            c1 = mappings[lab]
            for maps in merged[lab].keys():
                c2 = mappings[maps]
                confusion[c1][c2] = merged[lab][maps]
    return confusion

def mappingThreshold(mapCounts,threshold,limit):
    
    """
    Method to threshold the mappings at a specified frequencies.  Only those
    labels with mapping frequencies greater than the threshold will be
    included in the training model.
    
    Parameters:
    - - - - -
        mapCounts : dictionary of sub-dictionaries, where main keys
                        are labels, and sub-key/label pairs are labels
                        and an associated frequency
        threshold : count cutoff
        
        limit : whether to include labels above or below the threhsold
    Returns:
    - - - -
        passed : list of labels with frequencies greater than the cutoff
    """

    options = ['inside','outside']

    if limit not in options:
        raise ValueError('limit must be in {}.'.format(' '.join(options)))

    if threshold < 0:
        raise ValueError('threshold must be non-negative.')

    thresholdC = {k: [] for k in mapCounts.keys()}

    for key in mapCounts.keys():
        if mapCounts[key]:
            zips = zip(mapCounts[key].keys(),mapCounts[key].values())
    
            if limit == 'inside':
                passed = [k for k,v in zips if v <= threshold]
            else:
                passed = [k for k,v in zips if v >= threshold]
    
            thresholdC[key] = passed
        else:
            thresholdC[key] = None

    return thresholdC

def buildMappingMatrix(merged,R,*kwargs):
    
    """
    Method to build a binary matrix, where entries in this matrix correspond
    to labels that a vertex matches to.  Think of this matrix as the 
    adjacency matrix, but rather a mapping matrix.
    
    Parameters:
    - - - - -
        merged : vertex library for a single subject.
        R : the number of possible labels that were mapped to
    """
    
    if kwargs:
        if 'thresh' in kwargs.keys():
            T = kwargs['thresh']
    else:
        T = 0.05;
    
    mergedFreq = mappingFrequency(merged);
    
    mergedThresh = mappingThreshold(mergedFreq,T,'outside');
    
    N = len(mergedThresh.keys());
    
    mappingMatrix = np.zeros((N,R+1))
    
    for v in mergedThresh.keys():
        if mergedThresh[v]:
            maps = mergedThresh[v]
            mappingMatrix[v,maps] = 1;
            
    mappingMatrix = mappingMatrix[:,1:]
    
    return mappingMatrix;
            
