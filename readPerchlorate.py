import dataAnalyzerMN_IsoX as dA
import dataScreen_IsoX
import os
import json

allTriplicates = {}
#each experiment, or 'triplicate' is read separately as an .isox file
triplicateFileNames = [x for x in os.listdir('Perchlorate') if x.endswith('.isox')]
for thisFileName in triplicateFileNames:
    thisTriplicateObservation = 'Perchlorate/' + thisFileName

    #calculate output
    rtnAllFilesDF, mergedDict, allOutputDict = dA.calc_Folder_Output(thisTriplicateObservation,debug = True, cullByTime = True, scanNumber = False, timeBounds = (15,90))

    #rest use mergedList
    #Screen data for common problems
    dataScreen_IsoX.RSESNScreen(allOutputDict)
    dataScreen_IsoX.zeroCountsScreen(mergedDict)
    dataScreen_IsoX.internalStabilityScreenSubsequence(mergedDict,priorSubsequenceLength = 1000, testSubsequenceLength = 1000, thresholdConstant = 0.2)

    #Output as dictionary
    allTriplicates[thisFileName] = {'rtnAllFilesDF':rtnAllFilesDF,
                                    'mergedDict':mergedDict,
                                    'allOutputDict':allOutputDict}

    #.json output of main results
    with open('Perchlorate Results' + thisFileName + '.json', 'w', encoding='utf-8') as f:
        json.dump(allOutputDict, f, ensure_ascii=False, indent=4)