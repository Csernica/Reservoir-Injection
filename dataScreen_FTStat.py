##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Modified: July 29, 2022
Author: Tim Csernica
Contains the functions used to screen methionine data. 
"""

import os

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from tqdm import tqdm 
    
def RSESNScreen(allOutputDict):
    '''
    Screen all peaks and print any which have RSE/SN > 2

    Inputs:
        allOutputDict: A dictionary containing all of the output ratios, from dataAnalyzerMN.calc_Folder_Output

    Outputs:
        None. Prints flags for peaks that exceed the threshold. 
    '''

    for fragKey, ratioData in allOutputDict[0].items():
        for ratio in ratioData.keys():
            for fileIdx, fileData in enumerate(allOutputDict):

                RSESN = fileData[fragKey][ratio]['RelStError'] / fileData[fragKey][ratio]['ShotNoiseLimit by Quadrature']
                
                if RSESN >= 2:
                    print('File ' + str(fileIdx) + ' ' + fragKey + ' ' + ratio + ' fails RSE/SN Test with value of ' + str(RSESN))

def zeroCountsScreen(folderPath, fragmentDict, mergedList, fileExt = '.txt', threshold = 0):
    '''
    Iterates through all peaks and prints those with zero counts higher than a certain relative threshold.

    Inputs: 
        folderPath: The directory containing the .txt or .csv files from FTStatistic.
        fragmentDict: A dictionary containing information about the fragments and substitutions present in the FTStat output file. 
        mergedList: A list of lists; each outer list corresponds to a file, each inner list to a fragment; elements of this inner list are dataframes giving the scans and data for that fragment of that file. 
        fileExt: The file extension of the FTStat output file, either '.txt' or '.csv'
        threshold: The relative number of zero scans to look for

    Outputs:
        None. Prints the name of peaks with more than the threshold number of zero scans. 
    '''
    fileNames = [x for x in os.listdir(folderPath) if x.endswith(fileExt)]

    fragKeys = list(fragmentDict.keys())

    for fragKey, fragIsotopes in fragmentDict.items():
        fragIdx = fragKeys.index(fragKey)
        for iso in fragIsotopes:
            if iso != "OMIT":
                for fileIdx, fileName in enumerate(fileNames):
                    cDf = mergedList[fileIdx][fragIdx]
                    thisIsoFileZeros = len(cDf['counts' + iso].values) - np.count_nonzero(cDf['counts' + iso].values)
                    thisIsoFileZerosFraction = thisIsoFileZeros / len(cDf['counts' + iso])
                    if thisIsoFileZerosFraction > threshold:
                        print(fileName + ' ' + iso + ' ' + fragKey + ' has ' + str(thisIsoFileZeros) + ' zero scans, out of ' + str(len(cDf['counts' + iso])) + ' scans (' + str(thisIsoFileZerosFraction) + ')') 

def subsequenceOutlierDetection(timeSeries, priorSubsequenceLength = 1000, testSubsequenceLength = 1000):
    '''
    Calculates the anomaly score for a timeseries and a given subsequence length. 

    Inputs:
        timeSeries: A univariate time series, i.e. a pandas series.
        subsequenceLength: The length of the subsequence to use..

    Outputs:
        allDev: The euclidian distance between the subsequence of interest and the mean of the previous subsequenceLength observations.
    '''
    allDev = []
    for i in range(priorSubsequenceLength,len(timeSeries)-testSubsequenceLength):
        thisSubsequence = timeSeries[i:i+testSubsequenceLength]
        thisPrediction = timeSeries[i-priorSubsequenceLength:i].mean()
        meanZScore = np.abs(((thisSubsequence.values - thisPrediction) / thisSubsequence.std()).mean())
        allDev.append(meanZScore)

    return np.array(allDev)

def internalStabilityScreenSubsequence(folderPath, fragmentDict, fragmentMostAbundant, mergedList, MNRelativeAbundance = False, fileExt = '.txt', priorSubsequenceLength = 1000, testSubsequenceLength = 1000, thresholdConstant = 0.2):
    '''
    Screens all peaks for subsequence outlier detection; prints those that fail.

    Inputs: 
        folderPath: The directory containing the .txt or .csv files from FTStatistic.
        fragmentDict: A dictionary containing information about the fragments and substitutions present in the FTStat output file. 
        fragmentMostAbundant: A list giving the most abundant peak in each fragment.
        mergedList: A list of lists; each outer list corresponds to a file, each inner list to a fragment; elements of this inner list are dataframes giving the scans and data for that fragment of that file. 
        MNRelativeAbundance: Whether to calculate results as MN Relative Abundances or ratios.
        fileExt: The file extension of the FTStat output file, either '.txt' or '.csv'
        subsequenceLength: The length of the subsequence to use. 
        thresholdConstant: A constant to multiply by to determine the appropriate threshold for screening. 

    Outputs:
        No output; prints the identity of any file failing the test. 
    '''
    fileNames = [x for x in os.listdir(folderPath) if x.endswith(fileExt)]
    for fragKey, fragData in fragmentDict.items():
        fragIdx = list(fragmentDict.keys()).index(fragKey)
        for isoIdx, iso in enumerate(fragData):
            if iso != "OMIT":
                for fileIdx, fileData in tqdm(enumerate(mergedList)):
                    cDf = fileData[fragIdx]
                    if MNRelativeAbundance:
                        series = cDf['MN Relative Abundance ' + iso]
                    elif iso != fragmentMostAbundant[fragIdx]:
                        series = cDf[iso + '/' + fragmentMostAbundant[fragIdx]]
                    else:
                        continue
                    
                    allDev = subsequenceOutlierDetection(series, priorSubsequenceLength = priorSubsequenceLength, testSubsequenceLength = testSubsequenceLength)
                    if max(allDev) > thresholdConstant:
                        print("Failed Subsequence Detection " +  fileNames[fileIdx] + " " + fragKey + " " + iso + " with a value of " + "{:.2f}".format(max(allDev)))