import os
from re import S

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.signal import find_peaks
import dataAnalyzerMN_IsoX as dA
from tqdm import tqdm
  
    
def sortDf(dataFrame):
    try:
        dataFrame.index = [float(x) for x in dataFrame.index]
    except:
        pass
    dataFrame = dataFrame.sort_index(axis = 0)
    try:
        dataFrame.columns = [float(x) for x in dataFrame.columns]
    except:
        pass
    dataFrame = dataFrame.sort_index(axis = 1)
    
    return dataFrame

def peakDriftScreen(mergedDict, SmpStd = [], Replicate = [], driftThreshold = 2, textSize = 10, plot = False):
    subMassDict = {'d':1.00627674587,'15n':0.997034886,'13c':1.003354835,
                   'unsub':0,'33s':0.999387735,'34s':1.995795825,'36s':3.995009525,
                  '18o':2.0042449924,'17o':1.0042171364,'37cl':1.9970499,'37cl17o':1.9970499+1.0042171364,
                  '37cl18o':1.9970499+2.0042449924}

    peakDriftDict = {}
    firstKey = list(mergedDict.keys())[0]
    subNameList = mergedDict[firstKey]['subNameList']
    mostAbundantSub = dA.findMostAbundantSub(mergedDict[firstKey]['mergedDf'], subNameList)

    for iso in subNameList:
        if iso != mostAbundantSub:
            for fileIdx, (fileName, fileData) in enumerate(mergedDict.items()):
                thisMergedDf = fileData['mergedDf']
                observedMassIso = thisMergedDf[thisMergedDf['mass' + iso]!=0]['mass' + iso].mean()
                observedMassMostAbundant = thisMergedDf[thisMergedDf['mass' + mostAbundantSub]!=0]['mass' + mostAbundantSub].mean()

                computedMassMostAbundant = 0
                mASubs = mostAbundantSub.split('-')

                #Find the increase in mass due to substitutions
                for sub in mASubs:
                    try:
                        computedMassMostAbundant += subMassDict[sub.lower()]
                    except:
                        print("Could not look up substitution " + sub + " correctly.")
                        computedMassMostAbundant += 0

                computedMassIso = 0
                mSubs = iso.split('-')

                #Find the increase in mass due to substitutions
                for sub in mSubs:
                    try:
                        computedMassIso += subMassDict[sub.lower()]
                    except:
                        print("Could not look up substitution " + sub + " correctly.")
                        computedMassIso += 0

                #compute observed and theoretical mass differences
                massDiffActual = computedMassIso - computedMassMostAbundant
                massDiffObserve = observedMassIso - observedMassMostAbundant

                peakDrift = np.abs(massDiffObserve - massDiffActual)

                peakDriftppm = peakDrift / observedMassIso * 10**6

                if peakDriftppm > driftThreshold:
                    print("Peak Drift Observed for " + fileName + ' ' + " " + iso + " with size " + str(peakDriftppm))
                    
                if SmpStd[fileIdx] not in peakDriftDict:
                    peakDriftDict[SmpStd[fileIdx]] = {}

                peakDriftDict[SmpStd[fileIdx]][Replicate[fileIdx]] = peakDriftppm
                
            if plot:
                
                peakDriftDF = pd.DataFrame.from_dict(peakDriftDict).copy()

                fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4.5,3),dpi = 300)

                peakDriftDF = sortDf(peakDriftDF)

                sns.heatmap(peakDriftDF, ax = ax, annot=True, cmap = 'Blues',linewidths=.5, vmin = 0, 
                            vmax = 2, annot_kws={"size":textSize})
                ax.set_title(iso + ' Peak Drift')
    
def RSESNScreen(allOutputDict,  SmpStd = [], Replicate = [], textSize = 10, plot = False):
    firstFileName = list(allOutputDict.keys())[0]
    for fragKey, ratioData in allOutputDict[firstFileName].items():
        for ratio in ratioData.keys():
            for fileName, fileData in allOutputDict.items():

                RSESN = fileData[fragKey][ratio]['RelStError'] / fileData[fragKey][ratio]['ShotNoiseLimit']
                
                if RSESN >= 2:
                    print('File ' + fileName + ' ' + fragKey + ' ' + ratio + ' fails RSE/SN Test with value of ' + str(RSESN))

    if plot:
        for fragKey, fragData in allOutputDict[firstFileName].items():
            ratios = list(fragData.keys())
            for iso in ratios:
                RSESNDict = {}
                fileIdx = 0
                for fileName, fileData in allOutputDict.items(): 
                    for thisFragKey, thisFragment in fileData.items():
                        if thisFragKey == fragKey:
                            try:
                                SNRatio = thisFragment[iso]['RelStError'] / thisFragment[iso]['ShotNoiseLimit']
                            except:
                                SNRatio = 0
                                print("Could not find " + thisFragKey + ' ' + iso)
                            if SmpStd[fileIdx] not in RSESNDict:
                                RSESNDict[SmpStd[fileIdx]] = {}
                            RSESNDict[SmpStd[fileIdx]][Replicate[fileIdx]] = SNRatio
                    fileIdx += 1

                SnDf = pd.DataFrame.from_dict(RSESNDict).copy()

                fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4.5,3),dpi = 300)

                SnDf = sortDf(SnDf)

                sns.heatmap(SnDf, ax = ax, annot=True, cmap = 'Blues',linewidths=.5, vmin = 0, vmax = 2,
                                                     annot_kws={"size":textSize})
                ax.set_title(fragKey + ' ' + iso + ' SN Ratio')
                
def zeroCountsScreen(mergedDict, threshold = 0):
    firstKey = list(mergedDict.keys())[0]
    subNameList = mergedDict[firstKey]['subNameList']

    for iso in subNameList:
        for fileIdx, (fileName, fileData) in enumerate(mergedDict.items()):
            thisMergedDf = fileData['mergedDf']
            thisIsoFileZeros = len(thisMergedDf['counts' + iso].values) - np.count_nonzero(thisMergedDf['counts' + iso].values)

            thisIsoFileZerosFraction = thisIsoFileZeros / len(thisMergedDf['counts' + iso])
            if thisIsoFileZerosFraction > threshold:
                print(fileName + ' ' + iso + ' has ' + str(thisIsoFileZeros) + ' zero scans, out of ' + str(len(thisMergedDf['counts' + iso])) + ' scans (' + str(thisIsoFileZerosFraction) + ')') 


def internalStabilityScreen(mergedDict, MNRelativeAbundance = False, threshold = 0.3, N = 10):
    failedInternal = {}
    firstKey = list(mergedDict.keys())[0]
    subNameList = mergedDict[firstKey]['subNameList']
  
    for iso in subNameList: 
        print(iso)
        for fileIdx, (fileName, fileData) in enumerate(mergedDict.items()):
            thisMergedDf = fileData['mergedDf']
            mostAbundantSub = dA.findMostAbundantSub(thisMergedDf, subNameList)
            
            if MNRelativeAbundance:
                series = thisMergedDf['MN Relative Abundance ' + iso]
            elif iso != mostAbundantSub:
                series = thisMergedDf[iso + '/' + mostAbundantSub]
            else:
                continue

            l = len(series)
            iteration = l//N
            current = 0

            means = []
            serrs = []
            pVal = []
            for i in range(N):
                subseries = series[current:current+iteration]

                subMean = subseries.mean()
                subStd = subseries.std()
                serr = subStd / np.sqrt(iteration)
                rse = serr/subMean

                means.append(subMean)
                serrs.append(serr)

                seriesMinus = pd.concat([series[0:current],series[current+iteration:]])

                p = scipy.stats.ttest_ind(seriesMinus,subseries)[1]
                pVal.append(p)

                current += iteration
                
            target = (np.array(pVal) < 0.05).sum() / N

            if target > threshold:
                print("Failed Internal Stability " +  fileName + " " + iso + " " + str(target))

                failedInternal[iso].append(fileName)

    return failedInternal

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
    for i in tqdm(range(priorSubsequenceLength,len(timeSeries)-testSubsequenceLength)):
        thisSubsequence = timeSeries[i:i+testSubsequenceLength]
        thisPrediction = timeSeries[i-priorSubsequenceLength:i].mean()
        meanZScore = np.abs(((thisSubsequence.values - thisPrediction) / thisSubsequence.std()).mean())
        allDev.append(meanZScore)

    return np.array(allDev)

def internalStabilityScreenSubsequence(mergedDict, MNRelativeAbundance = False, priorSubsequenceLength = 1000, testSubsequenceLength = 1000, thresholdConstant = 0.2):
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
    firstKey = list(mergedDict.keys())[0]
    subNameList = mergedDict[firstKey]['subNameList']
  
    for iso in subNameList:
        print(iso)
        for fileIdx, (fileName, fileData) in enumerate(mergedDict.items()):
            thisMergedDf = fileData['mergedDf']

            mostAbundantSub = dA.findMostAbundantSub(thisMergedDf, subNameList)
            if MNRelativeAbundance:
                series = thisMergedDf['MN Relative Abundance ' + iso]
            elif iso != mostAbundantSub:
                series = thisMergedDf[iso + '/' + mostAbundantSub]
            else:
                continue

            allDev = subsequenceOutlierDetection(series, priorSubsequenceLength = priorSubsequenceLength, testSubsequenceLength = testSubsequenceLength)
            if max(allDev) > thresholdConstant:
                print("Failed Subsequence Detection " +  fileName + " " + iso + " with a value of " + "{:.2f}".format(max(allDev)))