##!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
#Author: Tim Csernica
#Last Updated: March 23, 2023

import copy
import itertools
import pandas as pd
import numpy as np

def readIsoX(fileName):
    '''
    Read in the 'combined' output from isoX; rename & drop columns as desired. 

    Inputs:
        fileName: A string; the name of the file ('combined.isox')

    Outputs:
        IsoXDf: A dataframe with the data from the .csv output. 
    '''
    IsoXDf = pd.read_csv(fileName, sep = '\t')

    IsoXDf.drop(columns=['ions.incremental','peakResolution','basePeakIntensity','rawOvFtT','intensCompFactor','agc','analyzerTemperature','numberLockmassesFound'], inplace = True)
    IsoXDf.rename(columns={'scan.no':'scanNumber','time.min':'retTime','it.ms':'integTime','mzMeasured':'mass'},inplace = True)
    #IsoX does not return TIC*IT by default. We can calculate it this way (divide by 1000 to convert from ms). The values calculated differ from the FTStat TIC*IT by <0.02%, in the files I checked. 
    IsoXDf['TIC*IT'] = IsoXDf['tic'] * IsoXDf['integTime'] / 1000

    return IsoXDf

def calculate_Counts_And_ShotNoise(peakDf,resolution=120000,CN=4.4,z=1):
    '''
    Calculate counts of each scan peak
    
    Inputs: 
        peakDf: An individual dataframe consisting of a single peak extracted by FTStatistic.
        CN: A factor from the 2017 paper to convert intensities into counts
        resolution: A reference resolution, for the same purpose (do not change!)
        z: The charge of the ion, used to convert intensities into counts
        Microscans: The number of scans a cycle is divided into, typically 1.
        
    Outputs: 
        The inputDF, with a column for 'counts' added. 
    '''
    peakDf['counts'] = (peakDf['intensity'] /
                  peakDf['peakNoise']) * (CN/z) *(resolution/peakDf['resolution'])**(0.5) * peakDf['microscans']**(0.5)
    return peakDf

def findMostAbundantSub(mergedDf, subNameList):
    '''
    Looks for the most abundant substituion in a dataframe and returns it. 

    Inputs:
        mergedDf: The dataframe of interest
        subNameList: A list of substitution names to search through

    Outputs:
        max_sub: A string, identifying the most abundant substitution.
    '''
    counts = []
    for sub in subNameList:
        thisCounts = mergedDf['counts'+sub].sum()
        counts.append(thisCounts)

    max_index = np.argmax(counts)
    max_sub = subNameList[max_index]

    return max_sub

def calc_Append_Ratios(mergedDf, subNameList, mostAbundant = True):
    '''
    Calculates the ratios for each combination of substitutions. Calculates all ratios in the order such that they are < 1.

    Inputs:
        mergedDf: A dataframe with all information for a single file. 
        subNameList: A list of substitution names, e.g. ['13C','18O','D']

    Outputs: 
        mergedDf: The same dataframe with ratios added. 
    '''
    max_sub = findMostAbundantSub(mergedDf, subNameList)

    for sub1, sub2 in itertools.combinations(subNameList,2):
        if ((mostAbundant) and (sub1 != max_sub) and (sub2 != max_sub)): 
            continue
        if mergedDf['counts' + sub1].sum() <= mergedDf['counts' + sub2].sum():
            mergedDf[sub1 + '/' + sub2] = mergedDf['counts' + sub1] / mergedDf['counts' + sub2]
        else:
            mergedDf[sub2 + '/' + sub1] = mergedDf['counts' + sub2] / mergedDf['counts' + sub1]
                            
    return mergedDf
    
def cull_By_Time(mergedDf, timeBounds, scanNumber = False):
    '''
    Takes a dataframe with orbitrap data and culls results outside the bounds of either scanNumber or retention time. 

    Inputs:
        mergedDf: A dataframe containing Orbitrap data
        timeBounds: A tuple containing the numerical values of the bounds
        scanNumber: If True, cull on scanNumber. Else, cull on retention time. 

    Outputs:
        mergedDf: The same dataframe with scans outside the bounds removed. 
    '''
    if scanNumber:
        mergedDf = mergedDf[mergedDf['scanNumber'].between(timeBounds[0], timeBounds[1], inclusive=True)]
    else:
        mergedDf = mergedDf[mergedDf['retTime'].between(timeBounds[0], timeBounds[1], inclusive=True)]
    return mergedDf
    
def combine_Substituted_Peaks(splitDf, cullByTime = False, scanNumber = False, timeBounds = (0,0)):
    '''
    A helper function for 'processIsoXDf'. Takes a Pandas groupby object containing dataframes for individual substitutions and combines them into a single dataframe for that acquisition.

    Inputs:
        splitDf: A Pandas groupBy object; can be iterated through; iterations give tuples of isotopolog strings ('D','M0', etc.) and dataframes corresponding to the data for that substitution. 
        cullByTime: If True, only include data within certain set of timepoints (by scan # or retTime)
        scanNumber: If True & cullByTime is True, then cull based on scan #. OTherwise, cull by retTime.
        timeBounds: A tuple; the retTime or scanNumber limits to use. 
    
    Outputs:
        combinedData: A dictionary with keys of subNameList (a list of substitutions in the dataframe) and mergedDf (a dataframe containing the combined information for all isotopes).
    '''
    combinedData = {}
    subIdx = 0 
    subNameList = []
    for subName, subData in splitDf:
        if subName == 'M0':
            subName = 'Unsub'
        subNameList.append(subName)
        subData = calculate_Counts_And_ShotNoise(subData)
        subData.rename(columns={'counts':'counts' + subName,'mass':'mass'+subName,'intensity':'intensity'+subName,'peakNoise':'peakNoise'+subName},inplace = True)

        if subIdx == 0:
            baseDf = copy.deepcopy(subData)
        else:
            subData.drop(columns=['filename','retTime','compound','isotopolog','tic','TIC*IT','integTime','resolution','agcTarget','microscans'],axis = 1, inplace=True)

            baseDf = pd.merge_ordered(baseDf, subData,on='scanNumber',suffixes =(False,False))

        subIdx += 1

    #If there are no entries for a given scan, IsoX will not include that scan in the output. 
    #This fills in 0s for all entries between the minimum and maximum scan
    maxScan = baseDf['scanNumber'].max()
    minScan = baseDf['scanNumber'].min()
    baseDf = baseDf.set_index('scanNumber').reindex(range(minScan,maxScan)).fillna(0).reset_index()

    #Fill in additional 0s
    for subName in subNameList:
        baseDf.loc[baseDf['mass' + subName].isnull(), 'mass' + subName] = 0
        baseDf.loc[baseDf['intensity' + subName].isnull(), 'intensity' + subName] = 0
        baseDf.loc[baseDf['peakNoise' + subName].isnull(), 'peakNoise' + subName] = 0
        baseDf.loc[baseDf['counts' + subName].isnull(), 'counts' + subName] = 0 

    baseDf = calc_Append_Ratios(baseDf, subNameList)

    if cullByTime: 
        baseDf = cull_By_Time(baseDf, timeBounds, scanNumber = scanNumber)
        
    combinedData['subNameList'] = subNameList
    combinedData['mergedDf'] = baseDf
    return combinedData

def processIsoXDf(IsoXDf, cullByTime = False, scanNumber = False, timeBounds = (0,0)):
    '''
    Takes in the IsoXDataframe; splits it based on filename. For each filename, combines the data from all substitutions into a single dataframe. Adds these dataframes to an output list. 

    Inputs:
        IsoXDf: A dataframe with the data from the .csv output.
        cullByTime: If True, culls results outside a certain set of time bounds (scanNumber or retTime)
        scanNumber: If True, the cullByTimes correspond to scan numbers. Else, they correspond to retention times. 
        timeBounds: A tuple giving the time bounds to cull on.

    Outputs:
        mergedDict: A dictionary where keys are file names and values are dictionaries; the keys of these are 'subNameList' (a list of substitutions for that dataframe) and 'mergedDf' (the actual dataframe). 
    '''
    filenames = list(set(IsoXDf.filename))
    filenames.sort()

    mergedDict = {}
    for thisFileName in filenames: 
        thisFileData = IsoXDf[IsoXDf['filename'] == thisFileName]

        #Check to see if there is any issue with multiple entries being found for the same substitution
        splitDf = thisFileData.groupby('isotopolog')
        for subName, subData in splitDf:
            #IsoX will return multiple values if multiple peaks are observed with closely similar masses. This prints a warning if this is the case and where. Not as useful as it could be, because low abundance (e.g. start of scan) periods will often have warnings as well. 
            g = splitDf.get_group(subName)['scanNumber'].value_counts()
            #if len(g[g>1]):
                #print("Multiple Entries Found for file " + thisFileName + " sub " + subName)
                #print(g[g>1])

        #For each isotopolog & each scan, select only the observation with highest intensity
        selectTopScan = thisFileData.loc[thisFileData.groupby(['isotopolog','scanNumber'])["intensity"].idxmax()]
        #sort by isotopolog to prepare for combine_substituted
        topScanDf = selectTopScan.groupby('isotopolog')

        thisCombined = combine_Substituted_Peaks(topScanDf, cullByTime = cullByTime, scanNumber = scanNumber, timeBounds = timeBounds)
        mergedDict[thisFileName] = thisCombined

    return mergedDict

def output_Raw_File_Ratios(mergedDf, subNameList, mostAbundant = True, massStr = None):
    '''
    For each ratio of interest, calculates mean, stdev, SErr, RSE, and ShotNoise based on counts. 
    Outputs these in a dictionary which organizes by fragment (i.e different entries for fragments at 119 and 109).
    
    Inputs:
        mergedDf: A dataframe containing information about a target set of isotopes.
        subNameList: A list of isotopes in the dataframe.
        mostAbundant: If true, only calculate ratios including the most abundant substitution
        massStr: Optional; set a value for the mass of the fragment in question. If left as none, the program will calculate it and fill it in. 
         
    Outputs: 
        A dictionary giving mean, stdev, StandardError, relative standard error, and shot noise limit for all isotope ratios.  
    '''
    #Initialize output dictionary 
    rtnDict = {}
      
    #Adds the peak mass to the output dictionary
    if massStr == None:
        #Try to set massStr programatically; first look for unsub
        try:
            massStr = str(round(mergedDf['massUnsub'].median(),1))
        #If this fails, find the first substitution and use that instead. 
        except:
            mergedEntries = list(mergedDf.keys())
            subNameList = [x[6:] for x in mergedEntries if 'counts' in x]
            massStr = str(round(mergedDf['mass' + subNameList[0]].median(),1))
        
    rtnDict[massStr] = {}
        
    maxSub = findMostAbundantSub(mergedDf, subNameList)

    #iterate through possible isotope ratios
    for sub1, sub2 in itertools.combinations(subNameList,2):
        if ((mostAbundant) and (sub1 != maxSub) and (sub2 != maxSub)): 
            continue
        if sub1 + '/' + sub2 in mergedDf:
            header = sub1 + '/' + sub2
        else:
            header = sub2 + '/' + sub1
          
        #perform calculations and add them to the dictionary     
        rtnDict[massStr][header] = {}
        rtnDict[massStr][header]['Ratio'] = np.mean(mergedDf[header])
        rtnDict[massStr][header]['StDev'] = np.std(mergedDf[header])
        rtnDict[massStr][header]['StError'] = rtnDict[massStr][header]['StDev'] / np.power(len(mergedDf),0.5)
        rtnDict[massStr][header]['RelStError'] = rtnDict[massStr][header]['StError'] / rtnDict[massStr][header]['Ratio']
        
        a = mergedDf['counts' + sub1].sum()
        b = mergedDf['counts' + sub2].sum()
        shotNoiseByQuad = np.power((1./a + 1./b), 0.5)
        rtnDict[massStr][header]['ShotNoiseLimit'] = shotNoiseByQuad

        averageTIC = np.mean(mergedDf['tic'])
        valuesTIC = mergedDf['tic']
        rtnDict[massStr][header]['TICVar'] = np.sqrt(
            np.mean((valuesTIC-averageTIC)**2))/np.mean(valuesTIC)

        averageTICIT = np.mean(mergedDf['TIC*IT'])
        valuesTICIT = mergedDf['TIC*IT']
        rtnDict[massStr][header]['TIC*ITMean'] = averageTICIT
        rtnDict[massStr][header]['TIC*ITVar'] = np.sqrt(
            np.mean((valuesTICIT-averageTICIT)**2))/np.mean(valuesTICIT)
                        
    return rtnDict

def calc_Folder_Output(isoXFileName, debug = False, cullByTime = False, scanNumber = False, timeBounds = (0,0)):
    '''
    Calculates the output for an input .isox file. The title 'calc_Folder_Output' corresponds to a folder of .RAW files, but a single .isox file. Returns a tuple with three data products of varying degrees of complexity. 

    Inputs:
        isoXFileName: A string, the name of the target .isox file
        debug: If true, prints file names
        cullByTime: If True, culls results outside a certain set of time bounds (scanNumber or retTime)
        scanNumber: If True, the cullByTimes correspond to scan numbers. Else, they correspond to retention times. 
        timeBounds: A tuple giving the time bounds to cull on.

    Outputs:
        rtnAllFilesDf: A dataframe containing a table with the output data
        mergedDict: A dictionary; keys are file names and values are dictionaries; then keys are 'subNameList' and 'mergedDf'; the mergedDfs contain the scan-by-scan data. 
        allOutputDict: A dictionary used to construct rtnAllFilesDf.
    '''
    IsoX = readIsoX(isoXFileName)
    mergedDict = processIsoXDf(IsoX, cullByTime = cullByTime, scanNumber = scanNumber, timeBounds = timeBounds)

    rtnAllFilesDF = []
    allOutputDict = {}
    #Go through by file and calculate output
    for thisFileName, thisFileData in mergedDict.items():
        if debug:
            print(thisFileName)

        thisMergedDf = thisFileData['mergedDf']
        thisSubNameList = thisFileData['subNameList']
        header = ["FileName", "Fragment", "IsotopeRatio", "Average", "StdDev", "StdError", "RelStdError",'ShotNoise']
        thisFileOutput = output_Raw_File_Ratios(thisMergedDf, thisSubNameList)

        #Fill in output dictionary
        allOutputDict[thisFileName] = thisFileOutput

        for fragKey, fragData in thisFileOutput.items():
            for subKey, subData in fragData.items():
            #add subkey to each separate df for isotope specific 
                thisRVal = subData["Ratio"]
                thisStdDev = subData["StDev"]
                thisStError = subData["StError"] 
                thisRelStError = subData["RelStError"]
                thisShotNoise = subData["ShotNoiseLimit"]
                thisRow = [thisFileName, fragKey, subKey, thisRVal, thisStdDev, thisStError,thisRelStError, thisShotNoise] 
                rtnAllFilesDF.append(thisRow)

    #Turn output dictionary into dataframe
    rtnAllFilesDF = pd.DataFrame(rtnAllFilesDF)

    # set the header row as the df header
    rtnAllFilesDF.columns = header 

    #sort by fragment and isotope ratio, output to csv
    rtnAllFilesDF = rtnAllFilesDF.sort_values(by=['Fragment', 'IsotopeRatio'], axis=0, ascending=True)
   
    return rtnAllFilesDF, mergedDict, allOutputDict

def folderOutputToDict(rtnAllFilesDF):
    '''
    Takes the output dataframe and processes to a dictionary, for .json output. Similar to allOutputDict; may be removed in a later version. 

    Inputs:
        rtnAllFilesDf: A dataframe containing a table with the output data

    Outputs:
        sampleOutputDict: A dictionary with processed output, suitable for export as a .json
    '''

    sampleOutputDict = {}
    for i, info in rtnAllFilesDF.iterrows():
        fragment = info['Fragment']
        file = info['FileName']
        ratio = info['IsotopeRatio']
        avg = info['Average']
        std = info['StdDev']
        stderr = info['StdError']
        rse = info['RelStdError']
        SN = info['ShotNoise']

        if file not in sampleOutputDict:
            sampleOutputDict[file] = {}

        if fragment not in sampleOutputDict[file]:
            sampleOutputDict[file][fragment] = {}

        sampleOutputDict[file][fragment][ratio] = {'Average':avg,'StdDev':std,'StdError':stderr,'RelStdError':rse,'ShotNoise':SN}
        
    return sampleOutputDict