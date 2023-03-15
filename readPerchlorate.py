import dataAnalyzerMN_IsoX as dA
import dataScreen_IsoX
import os
import json

allTriplicates = {}
triplicateFileNames = [x for x in os.listdir('Perchlorate') if x.endswith('.isox')]
for thisFileName in triplicateFileNames:
    thisTriplicateObservation = 'Perchlorate/' + thisFileName

    rtnAllFilesDF, mergedDict, allOutputDict = dA.calc_Folder_Output(thisTriplicateObservation,debug = True, cullByTime = True, scanNumber = False, timeBounds = (15,90))

    SmpStd = ['Std','Smp','Std','Smp','Std','Smp','Std']
    Replicate = ['1','1','2','2','3','3','4']
    #rest use mergedList

    dataScreen_IsoX.RSESNScreen(allOutputDict,SmpStd = SmpStd, Replicate = Replicate)
    dataScreen_IsoX.zeroCountsScreen(mergedDict)
    dataScreen_IsoX.internalStabilityScreenSubsequence(mergedDict,priorSubsequenceLength = 1000, testSubsequenceLength = 1000, thresholdConstant = 0.2)

    allTriplicates[thisFileName] = {'rtnAllFilesDF':rtnAllFilesDF,
                                    'mergedDict':mergedDict,
                                    'allOutputDict':allOutputDict}

    with open('Perchlorate Results' + thisFileName + '.json', 'w', encoding='utf-8') as f:
        json.dump(allOutputDict, f, ensure_ascii=False, indent=4)