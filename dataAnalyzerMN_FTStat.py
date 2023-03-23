##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Modified: July 29, 2022
Contains the functions used to read in and analyze FTStatistic output data. 
"""

import os
import copy
import math 

import numpy as np
import pandas as pd

#####################################################################
########################## FUNCTIONS ################################
#####################################################################

def import_Peaks_From_FTStatFile(inputFileName):
    '''
    Import peaks from FT statistic output file into a workable form, step 1
    
    Inputs:
        inputFileName: The FT Statistic output file to input from, either a .csv or a .txt.
        
    Outputs:
        A list, containing dictionaries for each mass with a set of peaks in the excel file. 
        The dictionaries have entries for 'tolerance', 'lastScan', 'refMass', and 'scans'. The 'scans' key directs to another list; 
        this has a dictionary for each indvidual scan, giving a bunch of data about that scan. 
    '''
    #Get data and delete header
    data = []
    for line in open(inputFileName):
        data.append(line.split('\t'))

    for l in range(len(data)):
        if data[l][0] == 'Tolerance:':
            del data[:l]
            break
    
    peaks = []
    n = -1
    
    for d in range(len(data)):
        if data[d][0] == 'Tolerance:':
            peaks.append({'tolerance': float(data[d][1].split()[0]),
                          'lastScan': int(data[d][7]),
                          'refMass': float(data[d][9]),
                          'scans': []})
            n += 1
        try:
            peaks[n]['scans'].append({'mass': float(data[d][1]),
                                      'retTime': float(data[d][2]),
                                      'tic': int(data[d][8]),
                                      'scanNumber': int(data[d][3]),
                                      'absIntensity': int(data[d][6]),
                                      'integTime': float(data[d][9]),
                                      'TIC*IT': int(data[d][10]),
                                      'ftRes': int(data[d][13]),
                                      'peakNoise': float(data[d][25]),
                                      'peakRes': float(data[d][27]),
                                      'peakBase': float(data[d][28])})
        except:
            pass
        
    return peaks

def convert_To_Pandas_DataFrame(peaks):
    '''
    Import peaks from FT statistic output file into a workable form, step 2
    
    Inputs:
        peaks: The peaks output from _importPeaksFromFTStatistic; a list of dictionaries. 
        
    Outputs: 
        A list, where each element is a pandas dataframe for an individual peak extracted by FTStatistic (i.e. a single line in the FTStat input .txt file). 
    '''
    rtnAllPeakDF = []

    for peak in peaks:
        try:
            columnLabels = list(peak['scans'][0])
            data = np.zeros((len(peak['scans']), len(columnLabels)))
        except:
            print("Could not find peak " + str(peak))
            continue
        # putting all scan data into an array
        for i in range(len(peak['scans'])):
            for j in range(len(columnLabels)):
                data[i, j] = peak['scans'][i][columnLabels[j]]
        # scan numbers as separate array for indices
        scanNumbers = data[:, columnLabels.index('scanNumber')]
        # constructing data frame
        peakDF = pd.DataFrame(data, index=scanNumbers, columns=columnLabels)

        # add it to the return pandas DF
        rtnAllPeakDF.append(peakDF)

    return(rtnAllPeakDF)

def calculate_Counts_And_ShotNoise(peakDF,resolution=120000,CN=4.4,z=1,Microscans=1):
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
    peakDF['counts'] = (peakDF['absIntensity'] /
                  peakDF['peakNoise']) * (CN/z) *(resolution/peakDF['ftRes'])**(0.5) * Microscans**(0.5)
    return peakDF

def calc_Append_Ratios(df, isotopeList = ['Unsub', '15N',  '13C']):
    '''
    Calculates all combinations of isotope ratios using the specified keys in the isotopeList. 
    Inputs:                               
            df: An individual pandas dataframe, consisting of multiple peaks from FTStat combined into one dataframe by the _combinedSubstituted function.
            isotopeList: A list of isotopes corresponding to the peaks extracted by FTStat for this fragment, in the order they were extracted. 

    Outputs:
            The dataframe with ratios added.
    '''
    for i in range(len(isotopeList)):
        for j in range(len(isotopeList)):
            if j>i:
                df[isotopeList[i] + '/' + isotopeList[j]] = df['counts' + isotopeList[i]] / df['counts' + isotopeList[j]]

    return df

def calc_MNRelAbundance(df, isotopeList = ['13C', '15N', 'Unsub']):
    '''
    An alternative way to calculate isotope ratios, as MN relative abundances rather than individual ratios. The denominator of a MN relative abundance is the sum of counts for all isotopes in the isotope list. 
    
    Inputs:
        df: An individual pandas dataframe, consisting of multiple peaks from FTStat combined into one dataframe by the _combinedSubstituted function.
        isotopeList: A list of isotopes corresponding to the peaks extracted by FTStat for this fragment, in the order they were extracted.

    Outputs:
        df: The same dataframe with the MN Relative abundances added. 
    '''
    df['total Counts'] = 0
    for sub in isotopeList:
        df['total Counts'] += df['counts' + sub]
        
    for sub in isotopeList:
        df['MN Relative Abundance ' + sub] = df['counts' + sub] / df['total Counts']
        
    return df

def combine_Substituted_Peaks(peakDF, cullOn = '', cullAmount = 3, onlySelectedTimes = False, selectedTimes = [], byScanNumber = False, fragmentIsotopeList = [['13C','15N','Unsub']], MNRelativeAbundance = False, Microscans = 1):
    '''
    Merge all extracted peaks from a given fragment into a single dataframe. For example, if I extracted six peaks, the 13C, 15N, and Unsub of fragments at mass 119 and 109, this would input a list of six dataframes (one per peak) and combine them into two dataframes (one per fragment), each including data from the 13C, 15N, and unsubstituted peaks of that fragment.
    
    Inputs: 
        peakDF: A list of dataframes. The list is the output of the _convertToPandasDataFrame function, and containts an individual dataframe for each peak extracted with FTStatistic. 
        cullOn: A target variable, like 'tic', or 'TIC*IT' to use to determine which criteria to cull on. 
        cullAmount: A number of standard deviations from the mean. If an individual scan has the cullOn variable outside of this range, culls the scan; e.g. if cullOn is 'TIC*IT' and cullAmount is 3, culls scans where TIC*IT is more than 3 standard deviations from its mean. 
        onlySelectedTimes: Set to True if you want to only analyze a subset of the dataframe
        selectedTimes: A list of 2-tuples. Each 2-tuple includes a starttime and an endtime to include for the corresponding fragment. 
        byScanNumber: select times based on scan number rather than retTime
        fragmentIsotopeList: A list of lists, where each interior list corresponds to a peak and gives the isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. This is used to determine all ratios of interest, i.e. 13C/Unsub, and label them in the proper order. 
        MNRelativeAbundance: If True, calculate M+N Relative abundances rather than isotope ratios.

    Outputs: 
        A list of combined dataframes; in the 119/109 example above, it will output a list of two dataframes, [119, 109] where each dataframe combines the information for each substituted peak.
    '''
    DFList = []
    peakIndex = 0
    
    thisTimeRange = []

    for fIdx, thisIsotopeList in enumerate(fragmentIsotopeList):
        #First substitution, keep track of TIC*IT etc from here
        df1 = peakDF[peakIndex].copy()
        sub = thisIsotopeList[0]
            
        if onlySelectedTimes == True:
            thisTimeRange = selectedTimes[fIdx]

        # calculate counts and add to the dataframe
        df1 = calculate_Counts_And_ShotNoise(df1, Microscans = Microscans)
       
        #Rename columns to keep track of them
        df1.rename(columns={'mass':'mass'+sub,'counts':'counts'+sub,'absIntensity':'absIntensity'+sub,
                            'peakNoise':'peakNoise'+sub},inplace=True)
        df1['sumAbsIntensity'] = df1['absIntensity'+sub]

        #add additional dataframes
        for additionalDfIndex in range(len(thisIsotopeList)-1):
            sub = thisIsotopeList[additionalDfIndex+1]
            df2 = peakDF[peakIndex + additionalDfIndex+1].copy()

            # calculate counts and add to the dataframe
            df2 = calculate_Counts_And_ShotNoise(df2, Microscans = Microscans)

            df2.rename(columns={'mass':'mass'+sub,'counts':'counts'+sub,'absIntensity':'absIntensity'+sub,
                            'peakNoise':'peakNoise'+sub},inplace=True)

            #Drop duplicate information
            df2.drop(['retTime','tic','integTime','TIC*IT','ftRes','peakRes','peakBase'],axis=1,inplace=True) 

            # merge with other dataframes from this fragment
            df1 = pd.merge_ordered(df1, df2,on='scanNumber',suffixes =(False,False))
            
        #Checks each peak for values which were not recorded (e.g. due to low intensity) and fills in zeroes
        for string in thisIsotopeList:
            df1.loc[df1['mass' + string].isnull(), 'mass' + string] = 0
            df1.loc[df1['absIntensity' + string].isnull(), 'absIntensity' + string] = 0
            df1.loc[df1['peakNoise' + string].isnull(), 'peakNoise' + string] = 0
            df1.loc[df1['counts' + string].isnull(), 'counts' + string] = 0 

        #Cull based on time frame
        if onlySelectedTimes == True and thisTimeRange != 0:
            df1= cull_On_Time(df1, timeFrame = thisTimeRange, byScanNumber = byScanNumber)

        #Calculates ratio values and adds them to the dataframe. Weighted averages will be calculated in the next step
        df1 = calc_Append_Ratios(df1, isotopeList = thisIsotopeList)
        
        if MNRelativeAbundance:
            df1 = calc_MNRelAbundance(df1, isotopeList = thisIsotopeList)

        #Given a key in the dataframe, culls scans outside specified multiple of standard deviation from the mean
        if cullOn != None:
            if cullOn not in list(df1):
                raise Exception('Invalid Cull Input')
            maxAllowed = df1[cullOn].mean() + cullAmount * df1[cullOn].std()
            minAllowed = df1[cullOn].mean() - cullAmount * df1[cullOn].std()

            df1 = df1.drop(df1[(df1[cullOn] < minAllowed) | (df1[cullOn] > maxAllowed)].index)

        peakIndex += len(thisIsotopeList)
        #Adds the combined dataframe to the output list
        DFList.append(df1)

    return DFList

def cull_On_Time(df, timeFrame = (0,0), byScanNumber = False):
    '''
    Select time window and only use scans that fall inside this time window. 

    Inputs: 
        df: input dataframe to cull
        timeFrame: Only use scans with retention times outside of this range, inclusive
        byScanNumber: Culls based on scanNumber, rather than retTime. 

    Outputs: 
       culled df based on input elution times for the peaks
    '''
    if byScanNumber == False: 
        # get the scan numbers for the retention  time frame
        if timeFrame != (0,0):
            #cull based on passed in retention time... 
            df = df[df['retTime'].between(timeFrame[0], timeFrame[1], inclusive=True)]
    
    else:
        if timeFrame != (0,0):
            df = df[df['scanNumber'].between(timeFrame[0], timeFrame[1], inclusive=True)]
    return df

def output_Raw_File_MNRelAbundance(df, fragKey = None, isotopeList = ['13C','15N','Unsub']):
    '''
    If you are doing MN Relative Abundances, use this function. For ratios, use calc_Raw_File_Output

    For each MN relative abundance of interest, calculates mean, stdev, SErr, RSE, and ShotNoise based on counts. 
    Outputs these in a dictionary which organizes by fragment (i.e different entries for fragments at 119 and 109).
    
    Inputs:
        df: A single merged dataframe from the _combineSubstituted function containing data about one fragment.
        fragKey: Can set the fragment string manually, otherwise it will be calculated from the data.
        isotopeList: A list of isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. 
        
    Outputs: 
        A dictionary with a single key, where the key corresponds to the fragment of interest (e.g. '109') and the value is a dictionaries giving mean, stdev, StandardError, relative standard error, and shot noise limit for all peaks.  
    '''
    #Initialize output dictionary 
    rtnDict = {}
      
    #Adds the peak mass to the output dictionary
    key = df.keys()[0]
    if fragKey == None:
        fragKey = str(round(df[key].median(),1))
        
    rtnDict[fragKey] = {}
    
    for sub in isotopeList:
        rtnDict[fragKey][sub] = {}
        rtnDict[fragKey][sub]['MN Relative Abundance'] = np.mean(df['MN Relative Abundance ' + sub])
        rtnDict[fragKey][sub]['StDev'] = np.std(df['MN Relative Abundance ' + sub])
        rtnDict[fragKey][sub]['StError'] = rtnDict[fragKey][sub]['StDev'] / np.power(len(df),0.5)
        rtnDict[fragKey][sub]['RelStError'] = rtnDict[fragKey][sub]['StError'] / rtnDict[fragKey][sub]['MN Relative Abundance']
        rtnDict[fragKey][sub]['TICVar'] = 0
        rtnDict[fragKey][sub]['TIC*ITVar'] = 0
        rtnDict[fragKey][sub]['TIC*ITMean'] = 0
                        
        a = df['counts' + sub].sum()
        b = df['total Counts'].sum()
        shotNoiseByQuad = np.power((1./a + 1./b), 0.5)
        rtnDict[fragKey][sub]['ShotNoiseLimit by Quadrature'] = shotNoiseByQuad
        
    return rtnDict        
                                              
def calc_Raw_File_Output(df, isotopeList = ['13C','15N','Unsub'], fragKey = None, removeZeroScans = False, omitRatios = [],  debug = True):
    '''
    If you are doing ratios, use this function. For MN Relative Abundances, use output_Raw_File_MNRelAbundance

    For each ratio of interest, calculates mean, stdev, SErr, RSE, and ShotNoise based on counts. Putputs these in a dictionary which organizes by fragment (i.e different entries for fragments at 119 and 109).
    
    Inputs:
        df: A single merged dataframe from the _combineSubstituted function containing data about one fragment.
        isotopeList: A list of isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. 
        fragKey: Can set the fragment string manually, otherwise it will be calculated from the data.
        removeZeroScans: If true, ignores zero scans when calculating these statistics. 
        omitRatios: A list of ratios to ignore. I.e. by default, the script will report 13C/15N ratios, which one may not care about, so these can be included in omitRatios and ignored. 
        debug: Set false to suppress print output
        
    Outputs: 
        A dictionary giving mean, stdev, StandardError, relative standard error, and shot noise limit for all peaks.  
    '''
    #Initialize output dictionary 
    rtnDict = {}
      
    #Adds the peak mass to the output dictionary
    key = df.keys()[0]
    if fragKey == None:
        fragKey = str(round(df[key].median(),1))
        
    rtnDict[fragKey] = {}
        
    for i in range(len(isotopeList)):
        for j in range(len(isotopeList)):
            if j>i:
                if isotopeList[i] + '/' + isotopeList[j] in df:
                    header = isotopeList[i] + '/' + isotopeList[j]
                else:
                    try:
                        header = isotopeList[j] + '/' + isotopeList[i]
                    except:
                        raise Exception('Sorry, cannot find ratios for your input isotopeList ' + header)

                if header in omitRatios:
                    if debug == True:
                        print("Ratios omitted:" + header)
                    continue
                else:
                    if removeZeroScans:
                        processedDf = df[(df['counts' + isotopeList[i]] != 0) & (df['counts' + isotopeList[j]] != 0)].copy()
                        print("Removing Zero Scans " + str(len(processedDf)))
                    else:
                        processedDf = df.copy()
                    #perform calculations and add them to the dictionary     
                    rtnDict[fragKey][header] = {}
                    rtnDict[fragKey][header]['Ratio'] = np.mean(processedDf[header])
                    rtnDict[fragKey][header]['StDev'] = np.std(processedDf[header])
                    rtnDict[fragKey][header]['StError'] = rtnDict[fragKey][header]['StDev'] / np.power(len(processedDf),0.5)
                    rtnDict[fragKey][header]['RelStError'] = rtnDict[fragKey][header]['StError'] / rtnDict[fragKey][header]['Ratio']
                    
                    a = processedDf['counts' + isotopeList[i]].sum()
                    b = processedDf['counts' + isotopeList[j]].sum()
                    shotNoiseByQuad = np.power((1./a + 1./b), 0.5)
                    rtnDict[fragKey][header]['ShotNoiseLimit by Quadrature'] = shotNoiseByQuad

                    averageTIC = np.mean(processedDf['tic'])
                    valuesTIC = processedDf['tic']
                    rtnDict[fragKey][header]['TICVar'] = math.sqrt(
                        np.mean((valuesTIC-averageTIC)**2))/np.mean(valuesTIC)

                    averageTICIT = np.mean(processedDf['TIC*IT'])
                    valuesTICIT = processedDf['TIC*IT']
                    rtnDict[fragKey][header]['TIC*ITMean'] = averageTICIT
                    rtnDict[fragKey][header]['TIC*ITVar'] = math.sqrt(
                        np.mean((valuesTICIT-averageTICIT)**2))/np.mean(valuesTICIT)
                        
    return rtnDict

def calc_Output_Dict(Merged, fragmentIsotopeList, fragmentMostAbundant, removeZeroScans = False, debug = True, MNRelativeAbundance = False, fragKeyList = None):
    '''
    For all peaks in the input file, calculates results via calc_Raw_File_Output and adds these results to a list. Outputs the final list. 
    
    Inputs:
        Merged: The list containing all merged data frames from the _combineSubstituted function. 
        fragmentIsotopeList: A list of lists, where each interior list corresponds to a peak and gives the isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. 
        fragmentMostAbundant: A list, where each entry is the most abundant isotope in a fragment. The order of fragments should correspond to the order given in "fragmentIsotopeList".  
        removeZeroScans: If True, calculates each ratio ignoring the zero scans for that ratio.
        debug: Set false to suppress print statements
        MNRelativeAbundance: Output MN Relative abundance rather than ratios. 
        fragKeyList: A list of fragment keys; the order should correspond to the fragment order in Merged. 
    
    Outputs:
        A list of dictionaries. Each dictionary has a single key value pair, where the key is the identity of the fragment and the value is a dictionary. The value dictionary has keys of isotope ratios (e.g. "D/13C") keyed to dictionaries giving information about that ratio measurement. 
        
        Note: Maybe rethink this as outputting a dictionary rather than a list, which may be cleaner? But outputting as a list keeps the same ordering as the original Merged list, which is convenient, while the fragment each entry corresponds to is clear.  
    '''
    outputDict = {}
    if fragKeyList == None:
        fragKeyList = [None] * len(fragmentIsotopeList)
    
    #branchpoint to use MNRelative abundances or ratios
    if MNRelativeAbundance:
        for fIdx, thisIsotopeList in enumerate(fragmentIsotopeList):
            output = output_Raw_File_MNRelAbundance(Merged[fIdx], fragKey = fragKeyList[fIdx], isotopeList = thisIsotopeList)
            fragKey = list(output.keys())[0]
            outputDict[fragKey] = output[fragKey]
            
    #ratios
    else:
        for fIdx, thisIsotopeList in enumerate(fragmentIsotopeList):
            mostAbundant = fragmentMostAbundant[fIdx]

            #calculate all possible permutations of isotopes, and omit any not including the most abundant isotope
            perms = []
            for x in thisIsotopeList:
                for y in thisIsotopeList:
                    perms.append(x + '/' + y)
            omitRatios = [x for x in perms if mostAbundant not in x.split('/')]

            output = calc_Raw_File_Output(Merged[fIdx],isotopeList = thisIsotopeList, fragKey = fragKeyList[fIdx], omitRatios = omitRatios, debug = debug, removeZeroScans = removeZeroScans)

            fragKey = list(output.keys())[0]
            outputDict[fragKey] = output[fragKey]
        
    return outputDict

def calc_Folder_Output(folderPath, cullOn=None, cullAmount=3, onlySelectedTimes=False, selectedTimes = [], fragmentIsotopeList = [['13C','15N','Unsub']], fragmentMostAbundant = ['Unsub'], debug = True, removeZeroScans = False, MNRelativeAbundance = False, fileExt = '.txt', fragKeyList = None, Microscans = 1):
    '''
    For each raw file in a folder, calculate mean, stdev, SErr, RSE, and ShotNoise based on counts. Outputs these in a dictionary which organizes by fragment. 
    Inputs:
        folderPath: Path that all the FTStatistic output files are in. Files must be in this format to be processed.
        cullOn: cull scans falling outside some standard deviation for this value, e.g. TIC*IT
        cullAmount: A number of standard deviations from the mean. If an individual scan has the cullOn variable outside of this range, culls the scan; i.e. if cullOn is 'TIC*IT' and cullAmount is 3, culls scans where TIC*IT is more than 3 standard deviations from its mean. 
        onlySelectedTimes: Specify whether you want to include only scans within a certain time window.
        selectedTimes: A list of tuples. Each tuple is the timeframe for a given fragment. 
        fragmentIsotopeList: A list of lists, where each interior list corresponds to a peak and gives the isotopes corresponding to the peaks extracted by FTStat, in the order they were extracted. 
        fragmentMostAbundant: A list, where each entry is the most abundant isotope in a fragment. The order of fragments should correspond to the order given in "fragmentIsotopeList".  
        debug: Set false to suppress print statements for omitted ratios and counts. 
        removeZeroScans: If True, calculates each ratio ignoring the zero scans for that ratio.
        MNRelativeAbundance: Output MN Relative abundance rather than ratios. 
        fileExt: the extension of the FTStatistic output files, either .csv or .txt.
        fragKeyList: A list of the fragment keys (e.g. '119'), matching the order of fragmentIsotopeList and fragmentMostAbundant.
        Microscans: The number of microscans to use for the calculation. 
               
    Outputs: 
        A 3-tuple, unpacked as:
        rtnAllFilesDF: A dataframe giving mean, stdev, standardError, relative standard error, and shot noise limit for all peaks. 
        mergedList: The list of merged dataframes, essentially the raw data prior to processing but in an accessible form.
        allOutputDict: A list of dictionaries where each element of the list corresponds to a file and the dictionary gives information about the fragments of that file. 

        The output is designed to give a broad range of products, so the user can select either the most developed (rtnAllFilesDF) or least developed (mergedList) as desired. 
    '''
    rtnAllFilesDF = []
    mergedList = []
    allOutputDict = []
    header = ["FileNumber", "Fragment", "IsotopeRatio", "Average", \
        "StdDev", "StdError", "RelStdError","TICVar","TIC*ITVar","TIC*ITMean", 'ShotNoise']
    #get all the file names in the folder with the same end 
    fileNames = [x for x in os.listdir(folderPath) if x.endswith(fileExt)]
    fileNames.sort()

    #Process through each raw file added and calculate statistics for fragments of interest
    for i in range(len(fileNames)):
        #initialize information, read in file.
        thisFragmentIsotopeList = copy.deepcopy(fragmentIsotopeList)
        thisFileName = str(folderPath + '/' + fileNames[i])
        print(thisFileName)
        thesePeaks = import_Peaks_From_FTStatFile(thisFileName)
        thisPandas = convert_To_Pandas_DataFrame(thesePeaks)

        #The user can omit substitutions that were present in the FTStat output file by writing "OMIT" in the fragment dictionary in place of the corresponding substitution. 
        FI = [sub for fragment in thisFragmentIsotopeList for sub in fragment]
        toOmit = []
        for i, sub in enumerate(FI):
            if sub == "OMIT":
                toOmit.append(i)
        for index in toOmit:
            thisPandas[index] = ''
            
        thisPandas = [p for p in thisPandas if type(p) != str]

        for i, isotopeList in enumerate(thisFragmentIsotopeList):
            thisFragmentIsotopeList[i][:] = [x for x in isotopeList if x != "OMIT"]
            
        #get combined dataframe for this file. 
        thisMergedDF = combine_Substituted_Peaks(peakDF=thisPandas,
                                                 cullOn=cullOn,
                                                 onlySelectedTimes=onlySelectedTimes, 
                                                 selectedTimes=selectedTimes, 
                                                 cullAmount=cullAmount, 
                                                 fragmentIsotopeList = thisFragmentIsotopeList, 
                                                 MNRelativeAbundance = MNRelativeAbundance,
                                                 Microscans = Microscans)
        
        mergedList.append(thisMergedDF)
        
        #analyze and output information from this file. 
        thisOutputDict = calc_Output_Dict(thisMergedDF, thisFragmentIsotopeList, fragmentMostAbundant, debug = debug, MNRelativeAbundance = MNRelativeAbundance, removeZeroScans = removeZeroScans, fragKeyList = fragKeyList)
        
        allOutputDict.append(thisOutputDict)

        #generate output dataframe
        for fragKey, fragData in thisOutputDict.items():
            for subKey, subData in fragData.items():
            #add subkey to each separate df for isotope specific 
                if MNRelativeAbundance:
                    thisRVal = subData["MN Relative Abundance"]
                else: 
                    thisRVal = subData["Ratio"]

                thisStdDev = subData["StDev"]
                thisStError = subData["StError"] 
                thisRelStError = subData["RelStError"]
                thisTICVar = subData["TICVar"] 
                thisTICITVar = subData["TIC*ITVar"]
                thisTICITMean = subData["TIC*ITMean"]
                thisShotNoise = subData["ShotNoiseLimit by Quadrature"]
                thisRow = [thisFileName, fragKey, subKey, thisRVal, \
                    thisStdDev, thisStError,thisRelStError,thisTICVar,thisTICITVar,thisTICITMean, thisShotNoise] 
                rtnAllFilesDF.append(thisRow)

    rtnAllFilesDF = pd.DataFrame(rtnAllFilesDF)

    # set the header row as the df header
    rtnAllFilesDF.columns = header 

    #sort by fragment and isotope ratio, output to csv
    rtnAllFilesDF = rtnAllFilesDF.sort_values(by=['Fragment', 'IsotopeRatio'], axis=0, ascending=True)
   
    return rtnAllFilesDF, mergedList, allOutputDict

def folderOutputToDict(rtnAllFilesDF):
    '''
    Takes the output dataframe and processes to a dictionary, for .json output.

    Inputs:
        rthAllFilesDF: The output dataframe from a folder analysis. 

    Outputs: 
        sampleOutputDict: An organized json file, where entries go file --> fragment --> ratio --> a dictionary containing information about that isotope ratio. This data product may be useful if sending the data to further scripts. 
    '''
    sampleOutputDict = {}
    for i, info in rtnAllFilesDF.iterrows():
        fragment = info['Fragment']
        file = info['FileNumber']
        ratio = info['IsotopeRatio']
        avg = info['Average']
        std = info['StdDev']
        stderr = info['StdError']
        rse = info['RelStdError']
        ticvar = info['TICVar']
        ticitvar = info['TIC*ITVar']
        ticitmean = info['TIC*ITMean']
        SN = info['ShotNoise']

        if file not in sampleOutputDict:
            sampleOutputDict[file] = {}

        if fragment not in sampleOutputDict[file]:
            sampleOutputDict[file][fragment] = {}

        sampleOutputDict[file][fragment][ratio] = {'Average':avg,'StdDev':std,'StdError':stderr,'RelStdError':rse,
                                   'TICVar':ticvar, 'TIC*ITVar':ticitvar,'TIC*ITMean':ticitmean,
                                  'ShotNoise':SN}
        
    return sampleOutputDict