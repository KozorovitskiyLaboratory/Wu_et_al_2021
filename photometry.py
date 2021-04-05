import os
import scipy.io as sio
import pandas as pd
import numpy as np


def siLatencyExtract(animal, condition, dataDir, dffFile, sweepLength):
    """function to get the latencies between scaniamge sweeps

    INPUTS:
        animal - (str) animal identifier
        condition - (str) the experimental condition
        dataDir - (str) path to the data
        dffFile - (str) the matlab file with all the dff data
        sweepLength - (float) the scanimage sweep length in seconds

    OUTPUT:
        siLatency - (list of floats) representing the latency (in seconds)
                    between the ith and i-1th scanimage sweep"""

    # get name of folder with raw scanimage files
    date = dffFile.name[:-4]

    # get contents of folder with raw scanimage files
    rawSiFiles = os.scandir(os.path.join(dataDir, animal, condition, date))

    tStamps = []
    for item in rawSiFiles:
        if item.name.endswith('.mat'):
            # load the file
            rawSi = sio.loadmat(item.path)

            # extract timestamp from raw scanimage file
            tStamps.append(float(rawSi[item.name[:-4]][0, 0][-3]))

    # sort the timestamps
    tStamps = sorted(tStamps)

    # get the scanimage latency length
    siLatency = [round((tStamps[i+1] - tStamps[i]) - sweepLength, 3)
                 for i in range(len(tStamps) - 1)]

    return siLatency


def dataEscExtractBeg(response, shock, dffData, xlnotes, samplingRate,
                      siLatency, sweepLength, esc10=False):
    """ function to extract a trial's dff Ca2+ transients 5s before 10s after
        beginning of shock

    INPUTS:
        response - (str) behavioral response
        shock - (int) the current shock number
        dffData - (dict with numpy arrays as values) the Ca2+ transient data
        xlnotes - (pandas dataframe) the behavioral data from Excel
        samplingRate - (int) the photometry sampling rate
        siLatency - (list of floats) representing the latency (in seconds)
                    between the ith and i-1th scanimage sweep
        sweepLength - (float) the scanimage sweep length in seconds
        esc10 - (bool) whether data is from the 10s long escapable shocks

    OUTPUTS:
        if response == 'premature':
            trialData - (np array) Ca2+ transient data for the given trial
        elif shockLength > 0:
            trialData
            shockLength - (float) the shock length in seconds
        else:
            np.nan
            shockLength"""

    # set first "Start Time" as reference
    if not esc10:
        refTime = xlnotes.loc[0, 'StartTime']
    else:
        refTime = xlnotes.loc[0, 'Start Time']

    if response == 'premature':
        # extract whole trial

        if not esc10:
            trialStart = (xlnotes.loc[shock, 'StartTime'] - refTime).seconds
            trialEnd = (xlnotes.loc[shock, 'EndTime'] - refTime).seconds
        else:
            trialStart = (xlnotes.loc[shock, 'Start Time'] - refTime).seconds
            trialEnd = (xlnotes.loc[shock, 'End Time'] - refTime).seconds

        # account for scanimage delay
        trialStart -= sum(siLatency[0:trialStart//sweepLength])
        trialEnd -= sum(siLatency[0:trialEnd//sweepLength])

        trialStart = round(trialStart)
        trialEnd = round(trialEnd)

        trialData = dffData[0, trialStart*samplingRate:trialEnd*samplingRate]

        return trialData

    else:
        # extract 5s before and 10s after shock begins

        # trial and shock ends at End Time
        # shock starts at Start Time + Total Pret + 5s

        if not esc10:
            trialStart = (xlnotes.loc[shock, 'StartTime'] - refTime).seconds
            pretrial = xlnotes.loc[shock, 'TotalPretrial']
            duration = xlnotes.loc[shock, 'Duration(s)']
        else:
            trialStart = (xlnotes.loc[shock, 'Start Time'] - refTime).seconds
            pretrial = xlnotes.loc[shock, 'Total Pretrial']
            duration = xlnotes.loc[shock, 'Duration(s)']

        shockLength = duration - pretrial - 5
        shockStart = int(trialStart + pretrial + 5)
        shockEnd = int(trialStart + duration)

        # account for scanimage delay
        shockEnd -= sum(siLatency[0:shockEnd//sweepLength])
        shockEnd = round(shockEnd)

        shockStart -= sum(siLatency[0:shockStart//sweepLength])
        shockStart = round(shockStart)

        if shockLength > 0:
            trialData = (
                dffData[
                    0, (shockStart-5)*samplingRate:(shockStart+10)*samplingRate
                    ]
                )
            return trialData, shockLength

        else:
            return np.nan, shockLength


def dataEscExtractEnd(response, shock, dffData, xlnotes, samplingRate,
                      siLatency, sweepLength, esc10=False):
    """ function to extract a trial's dff Ca2+ transients given the
    experiment condition +/- 5s around end of shock

    INPUTS:
        response - (str) behavioral response
        shock - (int) the current shock number
        dffData - (dict with numpy arrays as values) the Ca2+ transient data
        xlnotes - (pandas dataframe) the behavioral data from Excel
        samplingRate - (int) the photometry sampling rate
        siLatency - (list of floats) representing the latency (in seconds)
                    between the ith and i-1th scanimage sweep
        sweepLength - (float) the scanimage sweep length in seconds
        esc10 - (bool) whether data is from the 10s long escapable shocks

    OUTPUTS:
        if response == 'premature':
            trialData - (np array) Ca2+ transient data for the given trial
        elif shockLength > 0:
            trialData
            shockLength - (float) the shock length in seconds
        else:
            np.nan
            shockLength"""

    # set first "Start Time" as reference
    refTime = xlnotes.loc[0, 'StartTime']

    if response == 'premature':
        # extract whole trial

        trialStart = (xlnotes.loc[shock, 'StartTime'] - refTime).seconds
        trialEnd = (xlnotes.loc[shock, 'EndTime'] - refTime).seconds

        # account for scanimage delay
        trialStart -= sum(siLatency[0:trialStart//sweepLength])
        trialEnd -= sum(siLatency[0:trialEnd//sweepLength])

        trialStart = round(trialStart)
        trialEnd = round(trialEnd)

        trialData = dffData[0, trialStart*samplingRate:trialEnd*samplingRate]

        return trialData

    else:
        # extract 5s before and 5s after shock ends

        # trial and shock ends at End Time
        # shock starts at Start Time + Total Pret + 5s

        trialStart = (xlnotes.loc[shock, 'StartTime'] - refTime).seconds
        pretrial = xlnotes.loc[shock, 'TotalPretrial']
        duration = xlnotes.loc[shock, 'Duration(s)']
        shockStart = trialStart + pretrial + 5
        shockEnd = int(trialStart + duration)
        shockLength = shockEnd - shockStart

        # account for scanimage delay
        shockEnd -= sum(siLatency[0:shockEnd//sweepLength])
        shockEnd = round(shockEnd)

        if shockLength > 0:
            trialData = (
                dffData[0, (shockEnd-5)*samplingRate:(shockEnd+5)*samplingRate]
                )

            return trialData, shockLength

        else:
            return np.nan, shockLength


def windowEscData(animal, condition, alignType='beginning', keepOrder=False,
                  esc10=False, revision_saline=False, revision_control=False,
                  responses=False, revision_saline_daily=False):
    """ function to read dff Ca2+ transient data from escapeable trials and
        window it around each delivered shock

        INPUTS:
            animal - (str) animal identifier
            condition - (str) the experimental condition
            alignType - (str) whether to align traces to shock start or end.
                        This is set by passing 'beginning' or 'end'.
            keepOrder - (bool) whether to maintain the trial order and just
                        return a list of the trials' data in the order it
                        occured
            esc10 - (bool) denoting the data is from 10s long escapable shocks
            revision_saline - (bool) denoting whether to use the revision saline data
            revision_saline_daily - (bool) denoting whether to use the revision saline data that's split
                                    up by days instead of condition
            revision_control - (bool) denoting whether to use the revision control data
            responses - (bool) whether to return responses too

        OUTPUT:
            allData - (dict) keys: 'premature', 'escape', and 'failure'
                      for the traces from these behavioral responses. The
                      values are lists of lists with one sublist for each
                      trial. The first element in each trial's sublist
                      contains a string describing the behavioral response.
                      The behavioral responses are 'Premature', 'Escape', and
                      'No Escape' (failure). The second element is a numpy
                      array containing the Ca2+ transients. The escape and
                      failure sublists have a third element: a number
                      specifying the shock length in seconds.
            if keepOrder == True:
                orderedData - (list) containing the trials'
                              data in the order they occurred"""

    # Hz
    samplingRate = 250

    if revision_saline:
        dataDir = r'Learned_helplessness\data\revision\saline'
    elif revision_saline_daily:
        dataDir = r'Learned_helplessness\data\revision\saline_daily'
    elif revision_control:
        dataDir = r'Learned_helplessness\data\revision\control'
    elif esc10:
        dataDir = r'Learned_helplessness\data\escapable10s'
    else:
        dataDir = r'Learned_helplessness\data\escapable3s'

    # get dataDir contents for the given condition
    dataDirContents = os.scandir(os.path.join(dataDir, animal, condition))

    # get dff and shockbox output files
    dffFiles, xlFiles = [], []
    for item in dataDirContents:
        if item.name.endswith('.mat'):
            dffFiles.append(item)
        elif item.name.endswith('.xlsx'):
            xlFiles.append(item)

    # initialize list to store windowed data for all days of a given condition
    if not keepOrder:
        dataEscape, dataFailure, dataPremature = [], [], []
    elif keepOrder and not responses:
        orderedData = []
    elif keepOrder and responses:
        orderedWithResponses = []

    # loop through all the files within this condition's folder
    for dffFile in dffFiles:

        # load the file
        matFile = sio.loadmat(dffFile.path)

        # extract the data from the dict
        dffData = matFile['allDataDFF']

        # load the matching shockbox output from the condition's folder
        for xlItem in xlFiles:
            if ([int(i) for i in xlItem.name if i.isdigit()] ==
                [int(i) for i in dffFile.name if i.isdigit()]):

                xlnotes = pd.read_excel(xlItem.path)

        # convert start and end time columns to dateTime format
        if animal == 'B1' and not esc10 and not revision_control and not revision_saline:
            xlnotes.loc[:, 'StartTime'] = pd.to_datetime(
                xlnotes.loc[:, 'StartTime'],
                format=' %H:%M:%S')

            xlnotes.loc[:, 'EndTime'] = pd.to_datetime(
                xlnotes.loc[:, 'EndTime'],
                format=' %H:%M:%S')
        elif not esc10:
            # the output format was changed after animal B1's experiment
            xlnotes.loc[:, 'StartTime'] = pd.to_datetime(
                xlnotes.loc[:, 'StartTime'].map(lambda t: str(t)))

            xlnotes.loc[:, 'EndTime'] = pd.to_datetime(
                xlnotes.loc[:, 'EndTime'].map(lambda t: str(t)))
        elif esc10:
            xlnotes.loc[:, 'Start Time'] = pd.to_datetime(
                xlnotes.loc[:, 'Start Time'],
                format=' %H:%M:%S')

            xlnotes.loc[:, 'End Time'] = pd.to_datetime(
                xlnotes.loc[:, 'End Time'],
                format=' %H:%M:%S')

        print(xlnotes.shape[0])

        # get scanimage sweep length
        if animal[0] == 'B' and not esc10:
            sweepLength = 100
        elif esc10:
            sweepLength = 10
        elif revision_control or revision_saline:
            sweepLength = 10

        # get scanimage latencies
        siLatency = siLatencyExtract(animal, condition, dataDir, dffFile,
                                     sweepLength)

        # define the function to window the data
        if alignType == 'beginning':
            escExtract = dataEscExtractBeg
        elif alignType == 'end':
            escExtract = dataEscExtractEnd

        # conditionally format the No Response
        if not esc10:
            noResponse = 'NoResponse'
            totPretrial = 'TotalPretrial'
        else:
            noResponse = 'No Response'
            totPretrial = 'Total Pretrial'

        for shock in range(xlnotes.shape[0]):
            # assign each shock to one of three groups: 'premature'-
            # null, 'escape'- animal escaped, and 'no response'-
            # animal didn't escape (failure)

            if xlnotes.loc[shock, 'Avoidance'] == 1:
                if (xlnotes.loc[shock, totPretrial] + 5
                        < int(xlnotes.loc[shock, 'Duration(s)'])):

                    trialData, shockLength = escExtract(
                        'escape', shock, dffData, xlnotes, samplingRate,
                        siLatency, sweepLength, esc10)

                    if not keepOrder:
                        dataEscape.append(['Escape', trialData, shockLength])
                    elif keepOrder and not responses:
                        orderedData.append(trialData)
                    elif keepOrder and responses:
                        orderedWithResponses.append(['Escape', trialData])

                else:
                    if not keepOrder:
                        trialData = escExtract(
                            'premature', shock, dffData, xlnotes, samplingRate,
                            siLatency, sweepLength, esc10)

                        dataPremature.append(['Premature', trialData])

                    elif keepOrder and responses:
                        orderedWithResponses.append(['Premature', trialData])

            # 1. PreMature == 1
            elif xlnotes.loc[shock, 'PreMature'] == 1:
                trialData = escExtract(
                    'premature', shock, dffData, xlnotes, samplingRate,
                    siLatency, sweepLength, esc10)
                if not keepOrder:
                    dataPremature.append(['Premature', trialData])
                elif keepOrder and responses:
                    orderedWithResponses.append(['Premature', trialData])

            # 2. Escape == 1
            elif xlnotes.loc[shock, 'Escape'] == 1:
                trialData, shockLength = escExtract(
                    'escape', shock, dffData, xlnotes, samplingRate, siLatency,
                    sweepLength, esc10)

                if not keepOrder:
                    dataEscape.append(['Escape', trialData, shockLength])
                elif keepOrder and not responses:
                    orderedData.append(trialData)
                elif keepOrder and responses:
                    orderedWithResponses.append(['Escape', trialData])

            # 3. No Response == 1

            elif xlnotes.loc[shock, noResponse] == 1:
                trialData, shockLength = escExtract(
                    noResponse, shock, dffData,
                    xlnotes, samplingRate, siLatency, sweepLength, esc10)

                if not keepOrder:
                    dataFailure.append([noResponse, trialData, shockLength])
                elif keepOrder and not responses:
                    orderedData.append(trialData)
                elif keepOrder and responses:
                    orderedWithResponses.append(['Failure', trialData])

            else:
                print('error doesn\'t fall into any group')

    if not keepOrder:
        allData = {'premature': dataPremature, 'escape': dataEscape,
                   'failure': dataFailure}
        return allData
    elif keepOrder and not responses:
        return orderedData
    elif keepOrder and responses:
        return orderedWithResponses


def dataInescExtract(shock, dffData, xlnotes, samplingRate, siLatency,
                     sweepLength, trialLength=10):
    """ function to extract dff Ca2+ transients for inescapable trials

    INPUTS:
        shock - (int) the current shock number
        dffData - (dict with numpy arrays as values) the Ca2+ transient data
        xlnotes - (pandas dataframe) the behavioral data from Excel
        samplingRate - (int) the photometry sampling rate
        siLatency - (list of floats) representing the latency (in seconds)
                    between the ith and i-1th scanimage sweep
        sweepLength - (float) the scanimage sweep length in seconds
        trialLength - (int) length of trial in seconds. default = 10.

    OUTPUTS:
        trialData - (np array) Ca2+ transient data for the given trial"""

    # set first "Start Time" as reference
    refTime = xlnotes.loc[0, 'Start Time']

    # Shocks are all 3s. Extract 2s before shock begins and 5s after shock
    # ends for a total of 10s.
    # shock starts at Start Time + Total Pret + 1s

    # trial and shock ends at End Time
    # shock ends at end of trial (after duration amount of time)

    trialStart = (xlnotes.loc[shock, 'Start Time'] - refTime).seconds
    pretrial = xlnotes.loc[shock, 'Total Pretrial']
    shockStart = int(trialStart + pretrial + 1)

    shockStart -= sum(siLatency[0:shockStart//sweepLength])
    shockStart = round(shockStart)

    trialData = (dffData[0, (shockStart-2)*samplingRate:(shockStart +
                 trialLength-2) * samplingRate])

    return trialData


def windowInescData(animal, condition, trial_length=10):
    """ function to read dff Ca2+ transient data  from inescapable shock
        experiments and window it around each delivered shock

        INPUTS:
            animal - (str) animal identifier
            condition - (str) the experimental condition
            trialLength - (int) length of trial in seconds. default = 10.

        OUTPUT:
            rawData - (list of numpy arrays) each array is the dff data
                      from a single trial. A single trial is 10s long and
                      includes 2 seconds before the shock, the 3s shock, and
                      5s after."""

    # Hz
    samplingRate = 250

    # data directory on Sam's computer
    dataDir = r'Learned_helplessness\data\inescapable3s'

    # get dataDir contents for the given condition
    dataDirContents = os.scandir(os.path.join(dataDir, animal, condition))

    # get dff and shockbox output files
    dffFiles, xlFiles = [], []
    for item in dataDirContents:
        if item.name.endswith('.mat'):
            dffFiles.append(item)
        elif item.name.endswith('.xlsx'):
            xlFiles.append(item)

    # initialize list to store windowed data for all days of a given condition
    rawData = []

    # loop through all the files within this condition's folder
    for item in dffFiles:

        # load the file
        matFile = sio.loadmat(item.path)

        # extract the data from the dict
        dffData = matFile['allDataDFF']

        # load shockbox output from the condition's folder
        for xlItem in xlFiles:
            if xlItem.name[-6] == item.name[-5]:
                xlnotes = pd.read_excel(xlItem.path)

        # convert start and end time columns to dateTime format
        xlnotes.loc[:, 'Start Time'] = pd.to_datetime(
            xlnotes.loc[:, 'Start Time'],
            format=' %H:%M:%S')

        xlnotes.loc[:, 'End Time'] = pd.to_datetime(
            xlnotes.loc[:, 'End Time'],
            format=' %H:%M:%S')

        print(xlnotes.shape[0])

        # scanimage sweep length
        sweepLength = 10

        # get scanimage latencies
        siLatency = siLatencyExtract(animal, condition, dataDir, item,
                                     sweepLength)

        for shock in range(xlnotes.shape[0]):

            # extract data around each shock
            trialData = dataInescExtract(shock, dffData, xlnotes,
                                         samplingRate, siLatency, sweepLength,
                                         trialLength=trial_length)
            rawData.append(trialData)

    return rawData
