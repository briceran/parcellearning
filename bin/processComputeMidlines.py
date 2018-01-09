# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:41:24 2017

@author: kristianeschenburg
"""

import argparse,os,sys
sys.path.append('..')
sys.path.insert(0,'../../io/')

import loaded as ld

parser = argparse.ArgumentParser()
parser.add_argument('-hd','--homeDir',help='Home directory where data exists.',
                    required=True,type=str)
parser.add_argument('-sd','--subjectList',help='Subject list file.',
                    required=True,type=str)

args = parser.parse_args()

homeDir = args.homeDir
subjectList = args.subjectList
subjectList = ''.join([homeDir,subjectList])

with open(subjectList,'r') as inSubj:
    subjects = inSubj.readlines()
subjects = [x.strip() for x in subjects]

hemiMap = {'Left': 'L',
           'Right':'R'}

restPref = 'rfMRI_Z-Trans_merged_CORTEX_'

for subj in subjects:
    
    restDir = ''.join([homeDir,subj,'/RestingState/'])
    
    for h in hemiMap.keys():
        
        hemiDir = ''.join([restDir,h,'/'])
        
        if not os.path.exists(hemiDir):
            os.makedirs(hemiDir)
        
        oriRest = ''.join([restDir,restPref,h.upper(),'.gii'])
        newRest = ''.join([hemiDir,restPref,h.upper(),'.gii'])
        outMids = ''.join([hemiDir,subj,'.',hemiMap[h],'.Midline_Indices','.mat'])
        
        if os.path.exists(oriRest) and not os.path.exists(outMids):
            print oriRest
            print newRest
            print outMids
            ld.midline_restingState(oriRest,outMids,4800)
