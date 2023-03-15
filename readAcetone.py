##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Modified: July 29, 2022
Author: Tim Csernica
This code will read in the acetone data from .txt files and output a .json with the results.
"""
import dataAnalyzerMN_FTStat
import dataScreen_FTStat
import json

#Folder with the .txt files received from FTStatistic.
folderPath = "Acetone"
#This dictionary allows one to observe multiple fragments. Keys are fragment IDs and values are lists giving the substitutions observed for that fragment, in the order given in the FTStat file. The order for this dictionary should follow the FTStat file, i.e. the first key/value pair will correspond to the first n lines of the FTStat file.
fragmentDict = {'59':['13C-13C','18O','D','17O','13C','Unsub']}

#Choose the most abundant peak for each fragment. The order should be the same as the order of the key/value pairs in the above dictionary. This peak will be the denominator of the ratios for this fragment. 
fragmentMostAbundant = ['Unsub']

#Extract from dictionary. 
massStr = []
fragmentIsotopeList = []
for i, v in fragmentDict.items():
    massStr.append(i)
    fragmentIsotopeList.append(v)

###Set parameters for extraction 
#integrate only across the times included for each file. Here, 16-76 minutes of the .RAW file corresponds to our measurement (other timepoint correspond to purging, priming, etc.)
onlySelectedTimes = False
selectedTimes = [(0,0)]
#Any specific properties you want to cull on
cullOn = "TIC*IT"
#Multiple of SD you want to cull beyond for the cullOn property
cull_amount = 3
#Whether you want to calculate MN Relative Abundances (not used here); see the Mathematics paper. 
MNRelativeAbundance = False
#The File Extension of files to be read in
fileExt = '.txt'
#Note removeZeroScans removes zero scans when calculating the output, but all scans are retained in mergedList.
removeZeroScans = False

rtnAllFilesDF, mergedList, allOutputDict = dataAnalyzerMN_FTStat.calc_Folder_Output(folderPath, cullOn=cullOn, cullAmount=cull_amount,
                                               onlySelectedTimes=onlySelectedTimes, selectedTimes = selectedTimes, 
                                               fragmentIsotopeList = fragmentIsotopeList, 
                                               fragmentMostAbundant = fragmentMostAbundant, debug = False, 
                                               MNRelativeAbundance = MNRelativeAbundance, fileExt = fileExt, 
                                               fragKeyList = list(fragmentDict.keys()), removeZeroScans = removeZeroScans,
                                               Microscans = 10)

dataScreen_FTStat.RSESNScreen(allOutputDict)
dataScreen_FTStat.zeroCountsScreen(folderPath, fragmentDict, mergedList, fileExt = fileExt)
dataScreen_FTStat.internalStabilityScreenSubsequence(folderPath, fragmentDict, fragmentMostAbundant, mergedList, fileExt = fileExt)

sampleOutputDict = dataAnalyzerMN_FTStat.folderOutputToDict(rtnAllFilesDF)

with open('Acetone Results.json', 'w', encoding='utf-8') as f:
    json.dump(sampleOutputDict, f, ensure_ascii=False, indent=4)