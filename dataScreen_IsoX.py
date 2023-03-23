import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dataAnalyzerMN_IsoX as dA
from tqdm import tqdm
    
def RSESNScreen(allOutputDict):
    '''
    Screen all peaks and print any which have RSE/SN > 2

    Inputs:
        allOutputDict: A dictionary containing all of the output ratios, from dataAnalyzerMN.calc_Folder_Output

    Outputs:
        None. Prints flags for peaks that exceed the threshold. 
    '''
    firstFileName = list(allOutputDict.keys())[0]
    for fragKey, ratioData in allOutputDict[firstFileName].items():
        for ratio in ratioData.keys():
            for fileName, fileData in allOutputDict.items():

                RSESN = fileData[fragKey][ratio]['RelStError'] / fileData[fragKey][ratio]['ShotNoiseLimit']
                
                if RSESN >= 2:
                    print('File ' + fileName + ' ' + fragKey + ' ' + ratio + ' fails RSE/SN Test with value of ' + str(RSESN))
                
def zeroCountsScreen(mergedDict, threshold = 0):
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
    
    firstKey = list(mergedDict.keys())[0]
    subNameList = mergedDict[firstKey]['subNameList']

    for iso in subNameList:
        for fileIdx, (fileName, fileData) in enumerate(mergedDict.items()):
            thisMergedDf = fileData['mergedDf']
            thisIsoFileZeros = len(thisMergedDf['counts' + iso].values) - np.count_nonzero(thisMergedDf['counts' + iso].values)

            thisIsoFileZerosFraction = thisIsoFileZeros / len(thisMergedDf['counts' + iso])
            if thisIsoFileZerosFraction > threshold:
                print(fileName + ' ' + iso + ' has ' + str(thisIsoFileZeros) + ' zero scans, out of ' + str(len(thisMergedDf['counts' + iso])) + ' scans (' + str(thisIsoFileZerosFraction) + ')') 

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