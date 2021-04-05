import photometry
import numpy as np
import collections
from utils import simpleAxis
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as insLoc
from matplotlib.patches import Rectangle
import scipy.signal as ssig
import scipy.stats as sstats
import json
from itertools import zip_longest

# this file contains functions to conduct analyses and make plots.


def avgTraces(animals, condition, alignType, samplingRate, donut=False,
              sepDonut=False, showPlot=False, savePlot=False, saveSEM=False,
              esc10=False, revision_saline=False, revision_control=False,
              revision_saline_daily=False):
    """ function to plot the average traces for the 3 behavioral responses
        (Figure 2D)

        INPUTS:
            animals- (list) animal identifiers given as strings
            condition - (str) behavioral condition of interest
            alignType - (str) whether to align the traces to shock start or
                        end. This is set by passing 'beginning' or 'end'.
            samplingRate - (int) data sampling rate in Hz
            donut - (bool) whether to display a donut plot with percent of each
                    behavioral repsonse on avgTrace plot
            sepDonut - (bool) whether to save a separate donut plot with
                       percent of each behavioral repsonse
            showPlot - (bool) whether to show the plot
            savePlot - (bool) whether to save the plot
            saveSEM - (bool) whether to save the mean +/- SEM to text files
            esc10 - (bool) whether the data is from 10s long escapable shocks
            revision_saline - (bool) denoting whether to use the revision saline data
            revision_saline_daily - (bool) denoting whether to use the revision saline daily data
            revision_control - (bool) denoting whether to use the revision control data"""

    # get windowed data
    windowedData = [photometry.windowEscData(animal, condition, alignType,
                    esc10=esc10, revision_saline=revision_saline,
                    revision_control=revision_control,
                    revision_saline_daily=revision_saline_daily)
                    for animal in animals]

    # initialize list for the escape traces
    escapeTraces, failureTraces = [], []

    # initialize lists for escape shock lengths
    # To be used for color gradient in figure
    escapeShockLengths = []

    if not esc10:
        noResponse = 'NoResponse'
    else:
        noResponse = 'No Response'

    # split escape, no escape (failure), and premature data into separate lists
    for animalData in windowedData:
        for trial in animalData['escape']:

            # check whether shock length is > 0
            if trial[2] > 0 and trial[0] == 'Escape':
                escapeTraces.append(ssig.medfilt(trial[1], kernel_size=3))
                escapeShockLengths.append(trial[2])

        for trial in animalData['failure']:

            # check whether shock length is > 0
            if trial[2] > 0 and trial[0] == noResponse:
                failureTraces.append(ssig.medfilt(trial[1], kernel_size=3))

    prematureTraces = [trial[1] for trial in animalData['premature']
                       for animalData in windowedData]

    # initialize numpy arrays for average and SEMs of escape and failure

    # escape and failure trial lengths
    trialLengths = max(set([trace.shape[0] for trace in escapeTraces]))

    escapeAvg = np.zeros((trialLengths))
    escapeSEM = np.zeros(escapeAvg.shape)

    failureAvg = np.zeros((trialLengths))
    failureSEM = np.zeros(failureAvg.shape)

    # collect the dff values from a given timepoint for all traces
    for timeStep in range(trialLengths):
        temp = []

        for trial in escapeTraces:
            if trial.shape[0] == trialLengths:
                temp.append(trial[timeStep])

        if temp:
            # compute the mean and variance of the dff values
            escapeAvg[timeStep] = np.mean(temp)
            escapeSEM[timeStep] = sstats.sem(temp)

        temp = []
        # redo above for the failure trials
        for trial in failureTraces:
            if trial.shape[0] == trialLengths:
                temp.append(trial[timeStep])

        if temp:
            failureAvg[timeStep] = np.mean(temp)
            failureSEM[timeStep] = sstats.sem(temp)

    if saveSEM:
        fname = 'meanSEM_' + condition + '.txt'

        if len(animals) == 1:
            fname = '_'.join((animals[0], fname))

        with open(fname, 'w') as f:
            f.write('escape mean')
            f.write('\n')
            f.write(json.dumps(escapeAvg.tolist()))
            f.write('\n')
            f.write('escape SEM')
            f.write('\n')
            f.write(json.dumps(escapeSEM.tolist()))
            f.write('\n')
            f.write('failure mean')
            f.write('\n')
            f.write(json.dumps(failureAvg.tolist()))
            f.write('\n')
            f.write('failure SEM')
            f.write('\n')
            f.write(json.dumps(failureSEM.tolist()))

    if sepDonut:
        escapeColor = '#7570b3'
        failureColor = '#d95f02'

        plt.figure()
        plt.pie([len(escapeTraces), len(failureTraces)],
                colors=[escapeColor, failureColor], autopct='%1.0f%%',
                textprops=dict(color='k', fontsize=8),
                startangle=90, pctdistance=1.4,
                wedgeprops=dict(width=0.4, edgecolor='w'))
        plt.axis('equal')
        if len(animals) == 1:
            plt.savefig('_'.join((animals[0], condition, 'donut')))
        else:
            plt.savefig(condition+'_donut')

        plt.close()

    # PLOT
    if showPlot or savePlot:

        # bin escape shock lengths for gradient of shock depiction
        bins = collections.Counter(escapeShockLengths)

        # update each bin count to be itself + the sum of all bins with
        # greater keys
        for key in bins.keys():
            for key2 in bins.keys():
                if key2 > key:
                    bins[key] += bins[key2]

        # define y axis limits based on whether plotting multiple animals
        if len(animals) > 1:
            yPos = (-0.2, 0.201)
        else:
            yPos = (-0.2, 0.4)

        # ystep size
        yStep = 0.2

        # for each bin we need a tuple with the 1st value: the x value
        # for the left border of the rectangle 2nd value: rectangle width
        if alignType == 'beginning':
            rects = [(5, key) for key in bins.keys()]
            leftX = 5
        elif alignType == 'end':
            rects = [(5-key, key) for key in bins.keys()]
            leftX = 2

        # set the max alpha value
        alphaCeil = 0.3

        # set the bin with the most counts as the max alpha val
        maxCounts = max(bins.values())/alphaCeil

        # set the transparency values for each bin
        alphas = [val/maxCounts for val in bins.values()]

        # make array for x axes in seconds
        sec = np.linspace(0, trialLengths/samplingRate, trialLengths)

        # change figure size
        plt.rcParams['figure.figsize'] = [4, 11]

        # plot escape and failure avg traces
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

        # overall figure calls
        plt.xlim((0, trialLengths/samplingRate))
        plt.ylim(yPos[0], yPos[1])
        fig.suptitle(str(animals)+'_'+condition)

        # graph colors
        escapeColor = '#7570b3'
        failureColor = '#d95f02'
        shockColor = '#A9A9A9'

        # set SEM transparency
        semAlpha = 0.4

        # plot escape traces
        ax1.plot(sec, escapeAvg, color=escapeColor)
        ax1.set(title='Average of escape traces', ylabel='DF/F')
        ax1.fill_between(sec, escapeAvg+escapeSEM, escapeAvg-escapeSEM,
                         color=escapeColor, alpha=semAlpha,
                         linewidth=0.01)
        for i, rect in enumerate(rects):
            r = Rectangle((rect[0], yPos[0]), rect[1], yPos[1]-yPos[0],
                          alpha=alphas[i], color=shockColor)
            ax1.add_patch(r)
        ax1.annotate('n = ' + str(len(escapeTraces)),
                     xy=(12, 0.18))
        ax1.set_yticks(np.arange(yPos[0], yPos[1], yStep))
        ax1.set_yticks(np.arange(yPos[0], yPos[1], yStep))
        simpleAxis(ax1, 0)

        # plot failure traces
        ax2.plot(sec, failureAvg, color=failureColor)
        ax2.set(title='Average of failure traces', ylabel='DF/F')
        ax2.fill_between(sec, failureAvg+failureSEM, failureAvg-failureSEM,
                         color=failureColor, alpha=semAlpha,
                         linewidth=0.01)

        r = Rectangle((leftX, yPos[0]), 3, yPos[1]-yPos[0], alpha=alphaCeil,
                      color=shockColor)
        ax2.add_patch(r)
        ax2.annotate('n = '+str(len(failureTraces)),
                     xy=(12, 0.18))
        ax1.set_yticks(np.arange(yPos[0], yPos[1], yStep))
        ax2.set_yticks(np.arange(yPos[0], yPos[1], yStep))
        simpleAxis(ax2, 0)

        # plot escape and failure together
        ax3.plot(sec, escapeAvg, color=escapeColor)
        ax3.fill_between(sec, escapeAvg+escapeSEM, escapeAvg-escapeSEM,
                         color=escapeColor, alpha=semAlpha,
                         linewidth=0.01)
        ax3.plot(sec, failureAvg, color=failureColor)
        ax3.fill_between(sec, failureAvg+failureSEM, failureAvg-failureSEM,
                         color=failureColor, alpha=semAlpha,
                         linewidth=0.01)
        simpleAxis(ax3, 1)
        ax3.set_xticks(np.arange(0, 16, 5))
        ax3.set_xticklabels([str(val) for val in list(np.arange(0, 16, 5))])
        ax3.set_yticks(np.arange(yPos[0], yPos[1], yStep))

        if donut:
            # add donut plot to top left of graph to depict proportion of
            # trials in each condition
            inset = insLoc.inset_axes(ax3, width='20%', height='20%',
                                      loc='upper left')
            inset.pie([len(escapeTraces), len(failureTraces)],
                      colors=[escapeColor, failureColor], autopct='%1.0f%%',
                      textprops=dict(color='k', fontsize=8),
                      startangle=90, pctdistance=1.4,
                      wedgeprops=dict(width=0.4, edgecolor='w'))
            inset.axis('equal')

        # save and/or display plot based on function input
        if savePlot and len(animals) > 1:
            plt.savefig('all_'+condition+'_'+alignType
                        + '_filtered_avgTraces.tif')
            plt.close(fig)
        elif savePlot:
            plt.savefig((animals[0]+'_'+condition+'_'+alignType
                        + '_filtered_avgTraces.tif'))
            plt.close(fig)
        if showPlot:
            plt.show()


def getAvgData(animals, condition):
    """get trial-wise average of windowed data

        INPUTS:
            animals- (list) animal identifiers given as strings
            condition - (str) behavioral condition of interest

        OUTPUT:
            a numpy array with the number of columns equal to
            samplingRate*trialLength and rows equal to the number of trials.
            It contains the inescapable Ca2+ transient data saturated at 0.3"""
    if not isinstance(animals, list):
        raise AttributeError('animals must be a list')

    samplingRate = 250  # Hz
    trialLength = 10  # s

    rawData = [photometry.windowInescData(animal, condition)
               for animal in animals]

    # np.mean([a,b,c], axis=0)
    rawdata2 = [trial for trial in rawData
                if len(trial) == trialLength*samplingRate]

    rawdata2 = np.vstack(rawdata2)

    h = np.full(rawdata2.shape, 0.3)
    data = np.minimum(h, rawdata2)

    return data


def plotAvgTraces(strongToo=False, esc10=False, revision_saline=False,
                  revision_control=False, revision_saline_daily=False):
    """generate avg trace plots for escapable3s data

        INPUTS:
            strongToo - (bool) whether to include strongLH and strongKET data
            esc10 - (bool) whether the data is from 10s long escapable
                    shocks
            revision_saline - (bool) denoting whether to use the revision saline data
            revision_saline_daily - (bool) denoting whether to use the revision saline daily data
            revision_control - (bool) denoting whether to use the revision control data"""

    a = ['B1', 'B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
         'B14']
    conditions = ['NA', 'LH', 'KET']

    if esc10:
        a = ['A2', 'A3', 'A7', 'A9', 'A10', 'A11', 'A12']

    if revision_saline:
        a = ['B' + str(i) for i in range(1, 11, 1)]
        conditions = ['NA', 'LH', 'SALINE']

    if revision_saline_daily:
        a = ['B' + str(i) for i in range(12, 18, 1)]
        conditions = ['day' + str(i) for i in range(1, 7, 1)]

    if revision_control:
        a = ['A' + str(i) for i in range(1, 8, 1)]
        conditions = ['day' + str(i) for i in range(1, 7, 1)]

    if strongToo:
        conditions += ['STR_LH', 'STR_KET']
        a.remove('B8')

    for c in conditions:
        avgTraces(a, c, 'beginning', 250, sepDonut=True, savePlot=False,
                  esc10=False,
                  saveSEM=False,
                  revision_saline=revision_saline,
                  revision_control=revision_control,
                  revision_saline_daily=revision_saline_daily)


def modifyInescData(rawData, animal, condition, conciseEO=False,
                    conciseFL=False):
    """get windowed data

       rawData - (np array) calcium transient data with trials along the rows
       animal - (str) animal identifier
       condition - (str) condition of interest
       conciseEO - (bool) every other trial
       conciseFL - (bool) first 90 NA trials and last 90 LH trials"""

    samplingRate = 250  # Hz
    trialLength = 10  # s

    if conciseEO:
        # get every other trial
        rawdata2 = [trial for i, trial in enumerate(rawData)
                    if len(trial) == trialLength*samplingRate
                    and i % 2 == 0]
    elif conciseFL:
        # get first 90 of NA and last 90 of KET
        if condition == 'NA':
            # get first 90 NA
            rawdata2 = [trial for i, trial in enumerate(rawData)
                        if i <= 90
                        and len(trial) == trialLength*samplingRate]
        elif condition == 'LH':
            # get last 90 LH
            rawdata2 = [trial for i, trial in enumerate(rawData)
                        if i >= 90
                        and len(trial) == trialLength*samplingRate]
        else:
            # get all trials
            rawdata2 = [trial for trial in rawData
                        if len(trial) == trialLength*samplingRate]
    else:
        # get all trials
        rawdata2 = [trial for trial in rawData
                    if len(trial) == trialLength*samplingRate]

    rawdata2 = np.vstack(rawdata2)

    h = np.full(rawdata2.shape, 0.3)
    data = np.minimum(h, rawdata2)

    return data


def heatmap(data, animal, fileName, trialLength, shock):
    """plot all the traces for a given animal as a heatmap
       and save it as filename_animal

       INPUTS:
           data - (np array)
           animal - (str) animal identifier
           fileName - (str) prefix for the figure name
           trialLength - (int) length of the trial in seconds
           shock - (list) a length 2 list with values when the shock starts and
                   ends in seconds"""

    # data = np.vstack(data)
    cmap = 'RdBu_r'
    samplingRate = 250  # Hz

    yticks = list(range(30, len(data), 30))
    yticksLabels = [str(x) for x in yticks]

    fig, ax = plt.subplots(1, 1)
    c = ax.pcolormesh(data[::-1, :], cmap=cmap)
    ax.set_yticks(yticks)[::-1]
    ax.set_yticklabels(yticksLabels)
    ax.set_xticks(range(0, trialLength*samplingRate+1, samplingRate))
    ax.set_xticklabels(range(trialLength + 1))
    plt.xlabel('Time (s)')
    plt.ylabel('Trial')
    plt.title([animal, len(data)])
    fig.colorbar(c)

    # add vertical line depicting when shock occurs
    plt.plot([shock[0]*samplingRate, shock[0]*samplingRate],
             [0, len(data)], 'g--')
    plt.plot([shock[1]*samplingRate, shock[1]*samplingRate],
             [0, len(data)], 'g--')
    plt.savefig(fileName+'_'+animal)
    plt.close(fig)


def plotHeatmaps(inesc=False, strongToo=False, sep=False):
    """function to generate heatmaps of all traces from a given animal

       INPUTS:
           inesc - (bool) whether the data is from inescapable expts
           strongToo - (bool) whether to include strongLH and strongKET data
           sep - (bool) whether to separate the data by behavioral response
                 (only for escapable data)"""

    conditions = ['NA', 'LH', 'KET']

    if inesc and sep:
        raise ValueError('inesc and sep cannot both be True')

    if inesc:
        animals = ['A1', 'A2', 'A3', 'A7', 'A9', 'A10', 'A11', 'A12']
        trialLength = 10  # s
        shock = [2, 5]
        dffMax = 0.3
        expt = 'inesc'

    else:
        animals = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
        trialLength = 15
        shock = [5, 8]
        dffMax = 0.3
        expt = 'esc'

        if strongToo:
            animals.remove('B8')
            conditions += ['STR_LH', 'STR_KET']
            expt += '_strToo'

    samplingRate = 250  # Hz

    for animal in animals:

        dataAll = []

        # get data
        if inesc:
            for condition in conditions:
                dataAll.append(photometry.windowInescData(animal, condition))
        elif not sep:
            for condition in conditions:
                dataAll.append(photometry.windowEscData(animal, condition,
                               keepOrder=True))
        if dataAll:
            # only keep trials of correct length
            dataAll1 = [trial
                        for condition in dataAll
                        for trial in condition
                        if len(trial) == samplingRate*trialLength]

            # stack all the data in a np array
            data1 = np.vstack(dataAll1)

            # saturate the data at dffMax
            h = np.full(data1.shape, dffMax)
            dataAll2 = np.minimum(h, data1)

            # call the function which plots the heatmap
            heatmap(dataAll2, animal, 'all_'+expt, trialLength, shock)

        if sep:
            dffMin = -0.3
            for res in ['escape', 'failure']:
                dataAll = []

                for condition in conditions:
                    data = photometry.windowEscData(animal, condition)
                    for trial in data[res]:
                        dataAll.append(trial[1])

                # only keep trials of correct length
                dataAll1 = [ssig.medfilt(trial, kernel_size=5)
                            for trial in dataAll
                            if len(trial) == samplingRate*trialLength]

                # stack all the data in a np array
                data1 = np.vstack(dataAll1)

                # saturate the data at dffMax and dffMin
                h = np.full(data1.shape, dffMax)
                dataAll2 = np.minimum(h, data1)

                h = np.full(data1.shape, dffMin)
                dataAll2 = np.maximum(h, dataAll2)

                if res == 'escape':
                    # don't plot line depicting shock end
                    shock = [5, 5]
                else:
                    shock = [5, 5]

                # call the function which plots the heatmap
                heatmap(dataAll2, animal, 'all_'+res, trialLength, shock)


def save_inesc_data():
    """save inescapable data to txt"""

    conditions = ['NA', 'LH', 'KET']
    animals = ['A1', 'A2', 'A3', 'A7', 'A9', 'A10', 'A11', 'A12']
    trialLength = 20  # s
    samplingRate = 250  # Hz

    for animal in animals:

        dataAll = []

        # get data
        for condition in conditions:
            dataAll.append(photometry.windowInescData(animal, condition,
                           trial_length=trialLength))

        # only keep trials of correct length
        dataAll1 = [trial
                    for condition in dataAll
                    for trial in condition
                    if len(trial) == samplingRate*trialLength]

        if dataAll1:
            # stack all the data in a np array
            data1 = np.vstack(dataAll1)

            # save to txt
            np.savetxt(animal + '_inescapable_data.csv', data1, delimiter=",")
        else:
            print('no trials of length ' + str(trialLength))


def learningCurves(strongToo=False, together=False, plot=False, saveData=False,
                   returnLearning=False, esc10=False):
    """function to plot learning curves for Figure 2F

       INPUTS:
           strongToo - (bool) whether to include strongLH and strongKET data
           together - (bool) whether to plot all the animals' learning curves
                      together in one plot
           plot - (bool) whether to plot and save the learning curves
           saveData - (bool) whether to save the learning data in a text file
           returnLearning - (bool) whether to return the data
           esc10 - (bool) whether the data is from 10s long escapable shocks
           """

    animals = ['B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
               'B14']

    if esc10:
        animals = ['A2', 'A3', 'A7', 'A9', 'A10', 'A11', 'A12']

    conditions = ['NA', 'LH', 'KET']

    if strongToo:
        conditions += ['STR_LH', 'STR_KET']
        animals.remove('B8')

    if together:
        # initialize a list for learning curves from all animals
        allCurves = []

    for animal in animals:

        # keep a list of condition lengths for the x ticks
        lengths = []

        # list for learning curve
        y = []

        for c in conditions:
            # get responses
            responses = behavior.responseExtract(animal, c, esc10=esc10)

            if c == 'KET' and animal in ['B1', 'B2', 'B3', 'B4']:
                responses = responses[:-100]
            elif c == 'NA' and animal in ['B7', 'B8']:
                responses = responses[100:]

            for r in responses:
                if len(y) == 0:
                    if r == 'Escape':
                        y.append(1)
                    elif r == 'Failure':
                        y.append(-1)
                    else:
                        y.append(0)
                else:
                    if r == 'Escape':
                        y.append(y[-1] + 1)
                    elif r == 'Failure':
                        y.append(y[-1] - 1)
                    else:
                        y.append(y[-1] + 0)

            lengths.append(len(y))

        if not together and plot:
            label = [c+'\n'+str(cl) for c, cl in zip(conditions, lengths)]

            fig, ax = plt.subplots(1, 1)
            ax.plot(y)
            plt.xticks(lengths, label)
            plt.xlim((0, lengths[-1]))
            plt.yticks(range(-40, 81, 20))
            simpleAxis(ax, displayX=1)
            plt.title(animal)
            plt.savefig(animal+'_Learning')

        if together:
            allCurves.append(y)

    if together:
        if plot:
            # plot curves together
            label = [c+'\n'+str(cl) for c, cl in zip(conditions, lengths)]

            fig, ax = plt.subplots(1, 1)
            for curve, animal in zip(allCurves, animals):
                ax.plot(curve, label=animal)
            plt.xticks(lengths, label)
            plt.xlim((0, lengths[-1]))
            plt.legend()
            simpleAxis(ax, displayX=1)
            plt.savefig('all_Learning_together')

        if saveData:
            # save learning curves to text file
            with open('learningCurves.txt', 'w') as f:
                f.write(json.dumps(list(zip(animals, allCurves))))
                f.write(json.dumps(lengths))

        if returnLearning:
            return {animal: curve
                    for animal, curve in zip(animals, allCurves)}


def traceDistances(plot=False, saveDistances=False, saveMeanSEM=False,
                   saveNtrials=False, returnDistances=False):
    """function to compute the distance between a given animal's mean escape
       and failure traces. Saves a plot as a tif and the raw data to a txt
       for Figure 2G

       INPUTS:
           plot - (bool) whether to plot the data
           saveDistances - (bool) whether to save the distances
           saveMeanSEM - (bool) whether to save the mean and sem
           saveNtrials - (bool) whether to save the number of trials
           returnDistances - (bool) whether to return the distances"""

    conditions = ['NA', 'LH', 'KET']
    responses = ['escape', 'failure']

    animals = ['B1', 'B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
               'B13', 'B14']
    trialLength = 15
    samplingRate = 250  # Hz

    distance = {animal: [] for animal in animals}

    if saveMeanSEM:
        meanSEM = {animal: {res: {} for res in responses}
                   for animal in animals}

    if saveNtrials:
        nTrials = {animal: {res: [] for res in responses}
                   for animal in animals}

    for condition in conditions:
        for animal in animals:

            # get data
            data = {res: photometry.windowEscData(animal, condition)[res]
                    for res in responses}

            # only keep trials of correct length
            for res in responses:
                data[res] = [trial[1]
                             for trial in data[res]
                             if len(trial[1]) == samplingRate*trialLength]

            # stack all the data in a np array
            for res in responses:
                data[res] = np.vstack(data[res])

            # take the mean of all traces from each behavioral response
            escapeMean = np.mean(data['escape'], axis=0)
            failureMean = np.mean(data['failure'], axis=0)

            # compute some similarity measure between these means
            distance[animal].append(
                round(np.linalg.norm(escapeMean - failureMean), 2))

            if saveMeanSEM:
                for mean, res in zip([escapeMean, failureMean], responses):
                    meanSEM[animal][res]['mean'] = mean.tolist()
                    meanSEM[animal][res]['SEM'] = (
                        sstats.sem(data[res], axis=0).tolist())

            if saveNtrials:
                for res in responses:
                    nTrials[animal][res] = len(data[res])

        if saveMeanSEM:
            with open('meanSEM_'+condition+'.txt', 'w') as f:
                f.write(json.dumps(meanSEM))

        if saveNtrials:
            with open('nTrials_'+condition+'.txt', 'w') as f:
                f.write(json.dumps(nTrials))

    if plot:
        # plot distance
        x = [0, 1, 2]
        plt.rcParams['figure.figsize'] = [3, 6]
        fig, ax = plt.subplots(1, 1)
        for animal in animals:
            ax.plot([0, 1, 2], distance[animal], label=animal)
        plt.xticks(x, ['NA', 'LH', 'KET'])
        plt.ylabel('Escape & failure distance')
        plt.legend()
        simpleAxis(ax, displayX=1)
        plt.savefig('traceDistances')
        plt.close(fig)

    if saveDistances:
        # save distances to text file
        with open('traceDistances.txt', 'w') as f:
            f.write(json.dumps(distance))

    if returnDistances:
        return distance


def plotFeatures(plot=False, saveData=False):
    """compute time to peak and trough of each animal's mean trace as well as
       positive and negative AUC for figure S2 B

       INPUTS:
           plot - (bool) whether to plot the features
           saveData - (bool) whether to save the data"""

    conditions = ['NA', 'LH', 'KET', 'STR_LH', 'STR_KET']
    responses = ['escape', 'failure']

    animals = ['B1', 'B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10', 'B11']
    trialLength = 15
    samplingRate = 250  # Hz

    nFeatures = 5
    features = {feature: {animal: {res: []
                                   for res in responses}
                          for animal in animals}
                for feature in range(nFeatures)}

    for condition in conditions:
        for animal in animals:

            data = {res: [] for res in responses}

            # get data
            rawData = {res: photometry.windowEscData(animal, condition)[res]
                       for res in responses}

            # only keep trials of correct length
            for res in responses:
                for trial in rawData[res]:
                    if len(trial[1]) == samplingRate*trialLength:
                        data[res].append(trial[1])

            # stack all the data in a np array
            for res in responses:
                # take the mean of all traces from the given behavioral
                # response
                mean = np.mean(np.vstack(data[res]), axis=0)

                # compute peak
                features[0][animal][res].append(max(mean))

                # compute time to peak
                indx = np.where(mean == max(mean))
                features[1][animal][res].append(
                    (indx[0][0] / samplingRate) - 5)

                # compute time to trough
                indx = np.where(mean == min(mean[:12*samplingRate]))
                features[2][animal][res].append(
                    (indx[0][0] / samplingRate) - 5)

                # compute auc for trough
                meanB4 = np.mean(mean[:5*samplingRate])
                temp = [point for point in mean[5*samplingRate:12*samplingRate]
                        if point < meanB4]
                features[3][animal][res].append(np.trapz(temp))

                # compute auc for peak
                temp = [point for point in mean[5*samplingRate:]
                        if point > meanB4]
                features[4][animal][res].append(np.trapz(temp))

    if saveData:
        with open('features.txt', 'w') as f:
            f.write(json.dumps(features))

    if plot:
        feature = 0
        x = [0, 1, 2]
        fig, axs = plt.subplots(2, 4)
        for col in range(4):
            for row, res in enumerate(responses):
                for animal in animals:
                    axs[row, col].plot(x, features[feature][animal][res])
                axs[row, col].set_xticks(x, ['NA', 'LH', 'KET'])
                simpleAxis(axs[row, col], displayX=1)
            feature += 1
        plt.savefig('features_separate')
        plt.close(fig)

        fig, axs = plt.subplots(1, 4)
        for col in range(4):
            for animal in animals:
                axs[col].plot(x,
                              abs(np.array(features[col][animal]['escape']) -
                                  np.array(features[col][animal]['failure'])))
            axs[col].set_xticks(x, ['NA', 'LH', 'KET'])
            simpleAxis(axs[col], displayX=1)
        plt.savefig('features')
        plt.close(fig)


def grouper(iterable, n, fillvalue=[]):
    """Collect data into fixed-length chunks or blocks
       copied from Python documentation
       https://docs.python.org/3.7/library/itertools.html#itertools-recipes

       grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""

    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def inesc_latency(nAvg=10, plot=False, save=True):
    """Get the latency to peaks for calcium transients recorded during
       inescapable shocks. Latencies only returned for peaks > 3 standard
       deviations of pre-shock transient. Average every n trials

       INPUTS:
           nAvg - (int) how many trials to avg
           plot - (bool) whether to plot the data
           save - (bool) whether to save the data"""

    conditions = ['NA', 'LH', 'KET']
    animals = ['A1', 'A2', 'A7', 'A9', 'A10', 'A11', 'A12']

    samplingRate = 250
    trialLength = 10  # s
    shock = [2, 5]  # shock start and end time in seconds

    # make a dict for the latencies
    latencies = {condition: [] for condition in conditions}

    # count of groups that met the criteria or not
    n_met = 0
    n_failed = 0

    for condition in conditions:
        for animal in animals:

            # get the data
            data = photometry.windowInescData(animal, condition)

            if save:
                if nAvg:
                    for trials in grouper(data, nAvg):
                        lens = [
                            len(trial) == samplingRate*trialLength
                            for trial in trials]

                        if all(lens):
                            # average the group of trials
                            trial = np.mean(trials, axis=0)

                            # standard deviation before shock
                            std = np.std(trial[:shock[0]*samplingRate])

                            # dff peak amplitude after shock start
                            peak = max(trial[shock[0]*samplingRate:])

                            if peak > 3*std:
                                # latency from shock start to peak
                                indx = np.where(
                                    trial[shock[0]*samplingRate:] == peak)
                                latencies[condition].append(
                                    (indx[0][0] / samplingRate) + shock[0])

                                n_met += 1

                            else:
                                n_failed += 1

                else:
                    for trial in data:
                        if len(trial) == samplingRate*trialLength:
                            # standard deviation before shock
                            std = np.std(trial[:shock[0]*samplingRate])

                            # get the dff peak amplitude after shock
                            peak = max(trial[2*samplingRate:])

                            if peak > 3*std:
                                # get the latency from shock start to peak
                                indx = np.where(trial[2*samplingRate:] == peak)
                                latencies[condition].append(
                                    (indx[0][0] / samplingRate) + 2)
            if plot:
                if nAvg:
                    for trials in grouper(data, nAvg):
                        lens = [
                            len(trial) == samplingRate*trialLength
                            for trial in trials]

                        if all(lens):
                            trial = np.mean(trials, axis=0)

                            # get the dff peak amplitude after shock
                        peak = max(trial[2*samplingRate:])
                        latencies[condition][0].append(peak)

                        # get the latency from shock start to peak
                        indx = np.where(trial[2*samplingRate:] == peak)
                        latencies[condition][1].append(
                            (indx[0][0] / samplingRate) + 2)

                        # get the amplitude of the trough
                        trough = min(trial[2*samplingRate:5*samplingRate])
                        latencies[condition][2].append(trough)
                else:
                    for trial in data:
                        if len(trial) == samplingRate*trialLength:
                            # get the dff peak amplitude after shock
                            peak = max(trial[2*samplingRate:])
                            latencies[condition][0].append(peak)

                            # get the latency from shock start to peak
                            indx = np.where(trial[2*samplingRate:] == peak)
                            latencies[condition][1].append(
                                (indx[0][0] / samplingRate) + 2)

                            # get the amplitude of the trough
                            trough = min(trial[2*samplingRate:5*samplingRate])
                            latencies[condition][2].append(trough)

    if save:
        with open('latencyInescapable.txt', 'w') as f:
            f.write('%d groups of %d transients were 3 standard deviations above pre-shock transient and %d groups were below' % (n_met, nAvg, n_failed))
            f.write(' ')
            f.write(json.dumps(latencies))
