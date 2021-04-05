import os
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from utils import simpleAxis
import scipy.stats as sstats
import json

# functions to analyze the locomotion control data in Figure 1H right and S1 E


def transitionExtract(condition, animal):
    """extract the times when a transition between start and stop or the
    reverse occurs

    INPUTS:
        condition - (str) the experimental condition
        animal - (str) the animal identifier

    OUTPUTS:
        startTimes - (list) the transition times, in seconds, from not
                     moving to moving
        stopTimes - (list) the transition times, in seconds, from moving to
                    not moving"""

    # data directory on Sam's computer
    dataDir = (
        r'C:\Users\sam\Box\MZW_genia\Learned_helplessness\data\locomotion')

    # get dataDir contents for the given animal and condition
    dataDirContents = os.scandir(os.path.join(dataDir, animal, condition))

    # read instantaneous velcoity from tracker output into a Pandas dataframe
    for f in dataDirContents:
        if f.name.endswith('.xls'):
            data = pd.read_excel(f, sheet_name=1, header=None,
                                 usecols='BP', names=['velocity'],
                                 skiprows=[0, 1, 2, 3, 4])
            if data['velocity'].iloc[0] == 1:
                data = pd.read_excel(f, sheet_name=1, header=None,
                                     usecols='BQ', names=['velocity'],
                                     skiprows=[0, 1, 2, 3, 4])
            frameRateCol = pd.read_excel(f, sheet_name=0, header=None,
                                         usecols='B')

    temp = frameRateCol.values.tolist()
    frameRate = temp[-3][0]

    # add column with binary values denoting movement.
    # consider movement in time blocks of 0.5 seconds
    # and define block as stationary if the animal's velocity is < 30 mm/s
    # in every frame.

    block = frameRate // 2
    data['moving'] = 0
    for frame in range(0, data.shape[0]-block, block):
        if all(data['velocity'].loc[frame:frame+block] > 30):
            data.loc[frame:frame+block, 'moving'] = 1

    # add binary column with values denoting start
    data['start'] = 0
    for frame in range(0, data.shape[0]):
        if (data['moving'].iloc[frame] == 1
        and data['moving'].iloc[frame - 1] == 0):

            data.loc[frame, 'start'] = 1

    # add column with values denoting stop
    data['stop'] = 0
    for frame in range(0, data.shape[0]):
        if (data['moving'].iloc[frame] == 0
        and data['moving'].iloc[frame - 1] == 1):

            data.loc[frame, 'stop'] = 1

    # get the row indices where the animal transitioned
    start = data.index[data['start'] == 1].tolist()
    stop = data.index[data['stop'] == 1].tolist()

    # convert indices to times
    startTimes = [frame/frameRate for frame in start]

    stopTimes = [frame/frameRate for frame in stop]

    return startTimes, stopTimes


def dffExtract(condition, animal):
    """get the calcium transient data

    INPUTS:
        condition - (str) the experimental condition
        animal - (str) the animal identifier

    OUTPUTS:
        dffData - (list) the calcium transients"""

    # data directory on Sam's computer
    dataDir = (
        r'C:\Users\sam\Box\MZW_genia\Learned_helplessness\data\locomotion')

    # get dataDir contents for the given animal and condition
    dataDirContents = os.scandir(os.path.join(dataDir, animal, condition))

    # get dff data
    for f in dataDirContents:
        if f.name.endswith('.mat'):

            # load the file
            matFile = sio.loadmat(f.path)

            # extract the data from the dict
            data = matFile['allDataDFF'].tolist()

    return data[0]


def dffTransitions(condition, animals, window=5, plot=False, saveData=False):
    """get the calcium transient data around each movement transition
       (start -> stop and stop -> start)

    INPUTS:
        condition - (str) the experimental condition
        animal - (str) the animal identifier
        window - (int) the length of calcium transients, in seconds, to plot 
                  around each transition
        plot - (bool) whether to plot the windowed calcium transient data
        saveData - (bool) whether to save the calcium transient data to a text
                   file

    OUTPUTS:
        either figures or a text file with the average and SEM Ca2+
        transients around each motion onset/offset and the number of
        motion onsets/offsets."""

    # initialize lists for start and stop dff data
    startDff, stopDff = [], []

    for animal in animals:
        # get the transitions from the locomotion data
        startTimes, stopTimes = transitionExtract(condition, animal)

        # get calcium transient data
        dffData = dffExtract(condition, animal)

        # sampling rate of dff data (Hz)
        dffRate = 250

        # get dffData +/-window seconds around each transition

        # for transitions from not moving to moving
        for val in startTimes:

            # position of current transition in dff samples
            dffPos = val*dffRate

            # get +/- window seconds around the transition
            data = dffData[
                    int(dffPos-(window*dffRate)):
                    int(dffPos+(window*dffRate))]

            # only append data of correct length
            if len(data) == dffRate*2*window:
                startDff.append(data)

        for val in stopTimes:

            # position of current transition in dff samples
            dffPos = val*dffRate

            # get +/-window seconds around the transition
            data = dffData[
                    int(dffPos-(window*dffRate)):
                    int(dffPos+(window*dffRate))]

            # only append data of correct length
            if len(data) == dffRate*2*window:
                stopDff.append(data)

    # get the mean of all dff data at each timepoint
    startDffMean = [np.mean(timepoint) for timepoint in zip(*startDff)]
    stopDffMean = [np.mean(timepoint) for timepoint in zip(*stopDff)]

    # get the SEM of all dff data at each timepoint
    startDffSEM = [sstats.sem(timepoint) for timepoint in zip(*startDff)]
    stopDffSEM = [sstats.sem(timepoint) for timepoint in zip(*stopDff)]

    if saveData:
        with open('locomotionData_'+condition+'.txt', 'w') as f:
            f.write(json.dumps({'startDffMean': startDffMean,
                                'startDffSEM': startDffSEM,
                                'stopDffMean': stopDffMean,
                                'stopDffSEM': stopDffSEM,
                                'nStart': len(startDff),
                                'nStop': len(stopDff)}))

    if plot:
        # convert to numpy arrays to make plotting the SEM easier
        startDffMean = np.array(startDffMean)
        startDffSEM = np.array(startDffSEM)
        stopDffMean = np.array(stopDffMean)
        stopDffSEM = np.array(stopDffSEM)

        # plot calcium transients aligned to motion start times
        # make a list for the x axes
        sec = np.linspace(0, window*2, startDffMean.shape[0])

        fig, ax = plt.subplots()
        plt.plot(sec, startDffMean)
        plt.xlim((0, window*2))
        plt.fill_between(sec, startDffMean+startDffSEM,
                         startDffMean-startDffSEM,
                         alpha=0.4, linewidth=0.01)
        ax.set(title='Calcium transients aligned to motion start times',
               xlabel='Time (s)', ylabel='DF/F')
        ax.annotate('n = ' + str(len(startDff)), xy=(8.5, 0.09))
        simpleAxis(ax, displayX=1)
        plt.savefig('start_locomotion')
        plt.close(fig)

        # plot calcium transients aligned to motion stop times
        # make a list for the x axes
        sec = np.linspace(0, window*2, stopDffMean.shape[0])

        fig, ax = plt.subplots()
        plt.plot(sec, stopDffMean)
        plt.xlim((0, window*2))
        plt.fill_between(sec, stopDffMean+stopDffSEM,
                         stopDffMean-stopDffSEM,
                         alpha=0.4, linewidth=0.01)
        ax.set(title='Calcium transients aligned to motion stop times',
               xlabel='Time (s)', ylabel='DF/F')
        ax.annotate('n = ' + str(len(stopDff)), xy=(8.8, 0.09))
        simpleAxis(ax, displayX=1)
        plt.savefig('stop_locomotion')
        plt.close(fig)


def getDffTransitions():
    """master function to call dffTransitions"""
    conditions = ['NA', 'LH', 'KET']
    animals = ['A7', 'A9', 'A10', 'A11', 'A12']

    for condition in conditions:
        dffTransitions(condition, animals, saveData=True)
