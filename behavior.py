import os
import pandas as pd
import numpy as np
from utils import simpleAxis
from collections import Counter
from itertools import product, combinations, permutations
from graphviz import Digraph
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import chisquare
import math


def responseExtract(dataDir, animal, condition, daily=False, esc10=False,
                    esc10DataDir=None):
    """ function to get a list of the behavioral repsonses in the order they
        occured.

    INPUTS:
        dataDir - (str) path to data
        animal - (str) animal identifier
        condition - (str) behavioral condition
        daily - (bool) whether to keep the responses separate for each day
        esc10 - (bool) whether data is from the 10s long escapable shocks
        esc10DataDir - (str) path to esc10 data

    OUTPUTS:
        responses - (list) behavioral responses in the order they occured.
                    The options are 'Escape', 'Premature', and 'Failure'.
        if daily == True:
            responses
            dayLengths - (list) the number of trials in each day"""

    if not esc10:
        noResponse = 'NoResponse'
        totalPretrial = 'TotalPretrial'
    else:
        noResponse = 'No Response'
        totalPretrial = 'Total Pretrial'
        dataDir = esc10DataDir

    # get dataDir contents for the given condition
    dataDirContents = os.scandir(os.path.join(dataDir, animal, condition))

    # initialize list to store behavioral responses
    responses = []

    if daily:
        dayLengths = []

    # get shockbox output files
    xlFiles = [item for item in dataDirContents if item.name.endswith('.xlsx')]

    for xlFile in xlFiles:
        # read data from file
        xlnotes = pd.read_excel(xlFile)

        # go through each shock
        for shock in range(xlnotes.shape[0]):
            # assign each shock to one of three groups: 'premature':
            # null, 'escape': animal escaped, and 'no response'
            # animal didn't escape (failure)

            if xlnotes.loc[shock, 'Avoidance'] == 1:
                if (xlnotes.loc[shock, totalPretrial] + 5
                        < int(xlnotes.loc[shock, 'Duration(s)'])):
                    responses.append('Escape')

                else:
                    responses.append('Premature')

            # 1. PreMature == 1 (there was no shock, null comparison)
            elif xlnotes.loc[shock, 'PreMature'] == 1:
                responses.append('Premature')

            # 2. Escape == 1
            elif xlnotes.loc[shock, 'Escape'] == 1:
                responses.append('Escape')

            # 3. No Response == 1
            elif xlnotes.loc[shock, noResponse] == 1:
                responses.append('Failure')

            else:
                print('error doesn\'t fall into any group')

        if daily:
            dayLengths.append(len(responses))

    if daily:
        return responses, dayLengths
    else:
        return responses


def dailyResponses(animals, condition, response='Failure',
                   responseCounts=False):
    """function to get the daily percent of failures or response counts for a
       given animal

       INPUTS:
           animals - (list) with animal identifiers as strings
           condition - (str) behavioral condition
           response - (str) response of interest
           responseCounts - (bool) return daily count of each response

       OUTPUTS:
            pResponse- (dict) animal identifiers as keys and a list with
                        percent response for each day as values.
            if responseCounts:
                counts - (dict) with animal identifiers as keys and Counter
                         objects as values. The counter objects are counting
                         the number of occurences of each behavioral
                         response."""

    pResponse = {}
    counts = {}

    for animal in animals:
        responses, dayLengths = responseExtract(animal, condition, daily=True)

        # insert a 0 at the beginnning of dayLengths
        dayLengths.insert(0, 0)

        # split responses into each days' responses
        responses = [responses[length[0]:length[1]]
                     for length in zip(dayLengths, dayLengths[1:])]

        temp = []

        for i, day in enumerate(responses):
            c = Counter(day)

            if responseCounts:
                counts[animal+'_'+str(i)] = c

            if not responseCounts:
                temp.append(c[response] / sum(c.values()))

        pResponse[animal] = temp

    if not responseCounts:
        return pResponse

    if responseCounts:
        return counts


def getDailyResponses(res='Failure', strong=False, responseCounts=False,
                      save_data=False, return_data=False):
    """function to get the daily percent of a given response for the given
       animals and save them in a text file or return them

       INPUTS:
           res - (str) response of interest
           responseCounts - (bool) return the daily count of each response
           save_data - (bool) whether to save the data
           return_data - (bool) whether to return the data

       OUTPUTS:
           if return_data:
               responses - (list) of tuples where each tuple contains the
                           condition and the dict outputted by
                           dailyResponses."""

    animals = ['B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
               'B14']
    conditions = ['NA', 'LH', 'KET']
    fname = 'daily' + res + '.txt'

    if not responseCounts:
        responses = [(c, dailyResponses(animals, c, response=res))
                     for c in conditions]
    elif responseCounts:
        responses = [(c, dailyResponses(animals, c, response=res,
                                        responseCounts=True))
                     for c in conditions]

    if save_data:
        with open(fname, 'w') as f:
            f.write(res)
            f.write('\n')
            f.write(json.dumps(responses))

    if return_data:
        return responses


def responseProb(animals, condition, probability=False, counts=False):
    """computes the probability for each response

    INPUTS:
        animals - (list) containing animal identifiers as strings
        condition - (str) behavioral condition either 'NA', 'LH', or 'KET'
        probability - (bool) whether to return the probability of each
                      response (default: False)
        counts - (bool) whether to return the number of times each response
                 occured

    OUTPUTS:
        probabilities - (dict) responses as keys and response probability as
                               values
        counts - (Counter) counts of each response"""

    # create an empty countainer for the response pair counts
    count = Counter()

    for animal in animals:
        # get sequence of behavioral responses (it will be a list)
        responses = responseExtract(animal, condition)

        if condition == 'KET' and animal in ['B2', 'B3', 'B4']:
            responses = responses[:-100]
        elif condition == 'NA' and animal in ['B7', 'B8']:
            responses = responses[100:]

        # get dict with responses as keys and counts as vals *for all animals*
        count += Counter(responses)

    # response names
    resNames = count.keys()

    if probability:
        # get probability of each response
        return {r: round(count[r] / sum(count.values()), 2)
                for r in resNames}
    elif counts:
        return count


def transitionProb(animals, condition, nPairs=False):
    """computes a transition probability matrix from a sequence of behavioral
       events

    INPUTS:
        animals - (list) containing animal identifiers as strings
        condition - (str) behavioral condition either 'NA', 'LH', or 'KET'
        nPairs - (Bool) whether to return the count of each response pair

    OUTPUTS:
        tMatrix - (pandas dataframe) where the value at row i and column j
                  contains the probability the mouse transitioned from the
                  response at row i to the response at row j. The rows and
                  columns have labels: Escape, Failure, and Premature."""

    # create an empty countainer for the response pair counts
    count = Counter()

    for animal in animals:
        # get sequence of behavioral responses (it will be a list)
        responses = responseExtract(animal, condition)

        if condition == 'KET' and animal in ['B2', 'B3', 'B4']:
            responses = responses[:-100]
        elif condition == 'NA' and animal in ['B7', 'B8']:
            responses = responses[100:]

        # get all the pairs of consecutive responses
        responsePairs = zip(responses, responses[1:])

        # get dict with pairs as keys and counts as vals *for all animals*
        count += Counter(responsePairs)

    # response names
    resNames = list(set(responses))

    # initialize array for transition probabilities
    tMatrix_counts = pd.DataFrame(
        np.zeros((3, 3)), index=resNames, columns=resNames)

    tMatrix_prob = pd.DataFrame(
        np.zeros((3, 3)), index=resNames, columns=resNames)

    # fill in array with number of transitions from response i to j
    # where i is given by the row and j the column
    for response in resNames:
        for response1 in resNames:

            tMatrix_counts.at[response, response1] = count[
                response, response1
                ]

    # convert counts to probabilities
    for response in resNames:

        # compute sum of counts for a given response
        resCounts = sum(tMatrix_counts.loc[response])

        for response1 in resNames:
            tMatrix_prob.at[response, response1] = (
                round(tMatrix_prob.loc[response, response1] / resCounts, 2)
            )
    if not nPairs:
        return tMatrix_prob
    else:
        return tMatrix_counts, sum(count.values())


def plotStateTransitions(tMatrix, fName, fFormat='pdf'):
    """plot state transitions from a transition probability matrix

    INPUTS:
        tMatrix - (pandas dataframe) a transition probability matrix
                   with labeled axes
        fName - (str) filename for the graph
        fFormat - (str) file format for the graph (pdf, png)

    OUTPUTS:
        saves a graph in the current directory as well as a file of the same
        name with a .gv extension"""

    # create graph
    g = Digraph('marko', filename=fName+'.gv', format=fFormat)

    # change to horizontal layout
    g.attr(rankdir='LR')

    # get state names
    states = list(tMatrix)

    # add nodes
    for state in states:
        g.node(state, width='1.25', height='0.5', fixedsize='True')

    # get the pairs of states to use for edges. Note loops are allowed.
    pairs = product(states, repeat=2)

    # add edges and label each with probability of moving along it
    for pair in pairs:
        prob = tMatrix.loc[pair[0], pair[1]]
        g.edge(pair[0], pair[1],
               fontsize='8', label=str(prob),
               penwidth=str(2*prob), arrowsize=str(2*prob))

    # render graph
    g.view()


def generateGraphs(strongToo=False):
    """generates the state transition graphs

       INPUTS:
           strongToo - (bool) whether to include strongLH and strongKET data"""

    animals = ['B2', 'B3', 'B4', 'B7', 'B9', 'B10', 'B11', 'B12', 'B13',
               'B14']
    conditions = ['NA', 'LH', 'KET']

    if strongToo:
        conditions += ['STR_LH', 'STR_KET']
        animals.remove('B8')

    for condition in conditions:
        tMatrix = transitionProb(animals, condition)
        fName = '_'.join((condition, 'stateTransitions'))
        plotStateTransitions(tMatrix, fName)


def norm(a, b):
    """computes the frobenius norm of the difference between a and b"""
    return np.linalg.norm(a-b, ord='fro')


def computeSimilarity(strongToo=False):
    """computes the similarity between each pair of transition matrices.
       The values are saved into a text file transitionSimilarity.txt

       INPUTS:
           strongToo - (bool) whether to include the strongLH and strongKET
                       data
       OUTPUTS:
           a text file, transitionSimilarity.txt, with the similarity between
           matrix pairs"""

    animals = ['B2', 'B3', 'B4', 'B7', 'B9', 'B10', 'B11', 'B12', 'B13',
               'B14']
    conditions = ['NA', 'LH', 'KET']

    if strongToo:
        conditions += ['STR_LH', 'STR_KET']
        animals.remove('B8')

    # map between conditions and their transition probability matrices
    tMatrices = {condition:
                 transitionProb(animals, condition)
                 for condition in conditions}

    # get combinations of conditions
    pairs = list(combinations(conditions, 2))

    # compute similarity between pairs of conditions
    similarity = {pair[0]+'_'+pair[1]:
                  1 - norm(tMatrices[pair[0]], tMatrices[pair[1]])
                  for pair in pairs}

    # save the dict to a text file
    with open('transitionSimilarity.txt', 'w') as f:
        f.write(json.dumps(similarity))


def responseSequences(animals, length, esc10=False):
    """get response sequences of length length

    INPUTS:
        animals - (list) animals to consider
        length - (int) the sequence length to consider
        esc10 - (bool) whether this is for 10s escapable data

    OUTPUTS:
        p - (dict) keys: the sequences as a string joined by and underscore,
            vals: their occurence probability"""

    conditions = ['NA', 'LH', 'KET']
    possible_responses = ['Escape', 'Failure', 'Premature']

    # possible sequences
    possible_sequences = list(product(possible_responses,
                                      repeat=length))

    for i, condition in enumerate(conditions):

        counts = Counter()

        for animal in animals:

            # get responses
            responses = responseExtract(animal, condition, esc10=esc10)

            # get sequences of length steps
            sequences = zip(*(responses[i:] for i in range(length)))

            # get the counts of each sequence
            counts += Counter(sequences)

        # total of all sequences
        total_sequences = sum(counts.values())

        if i == 0:
            p = {}
            # place names and probabilities in dict
            for seq in possible_sequences:
                try:
                    p['_'.join(seq)] = [
                        round(counts[seq]/total_sequences, 2)]
                except KeyError:
                    p['_'.join(seq)] = [0]
        else:
            # place names and probabilities in dict
            for seq in possible_sequences:
                p['_'.join(seq)].append(
                    round(counts[seq]/total_sequences, 2))

    # remove items that never have nonzero probability
    p = {key: val for key, val in p.items() if sum(val) != 0}

    return p


def getResponseSequences(together=False, max_length=1):
    """save the response sequences of length max_length and their occurence
       probability in a text file

       INPUTS:
           together - (bool) whether to combine the data from all animals
           max_length - (int) the max sequence length to consider"""

    animals = ['B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
               'B14']

    if not together:
        for animal in animals:
            with open(animal + '_responseSequences.txt', 'w') as f:
                for i in range(1, max_length):
                    f.write('{} \n'.format(i))
                    f.write(json.dumps(responseSequences([animal], i)))
                    f.write('\n')
    else:
        with open('all_responseSequences.txt', 'w') as f:
            for i in range(1, max_length):
                f.write('{} \n'.format(i))
                f.write(json.dumps(responseSequences(animals, i)))
                f.write('\n')


def plotResponseSequences(together=False, response=None, max_length=1,
                          combine=False, esc10=False):
    """Plot the probability of successive escapes or failures. Up to
       max_length successive responses. Saves the plot and raw data

       INPUTS:
           together - (bool) whether to combine the data from all animals
           response - (str) the response of interest
           max_length - (int) the max sequence length to consider
           combine - (bool) whether to combine probabilities across conditions
           """

    animals = ['B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
               'B14']

    if esc10:
        animals = ['A2', 'A3', 'A7', 'A9', 'A10', 'A11', 'A12']

    conditions = ['NA', 'LH', 'KET']

    vals = []

    lengths = list(range(1, max_length+1))

    if combine:
        for length in lengths:
            response_seq = '_'.join([response]*length)

            # get probabilities
            p = responseSequences(animals, length, esc10=esc10)

            # get successive response values
            vals.append(p[response_seq])

        # combine probabilities from each condition
        vals = list(zip(*vals))

        # should add call to simpleAxis

        plt.figure()
        for i, val in enumerate(vals):
            plt.plot(lengths, val, label=conditions[i])
        plt.legend()
        plt.xlabel('Number of succesive responses')
        plt.ylabel('Probability of occuring')
        plt.title('_'.join(('combined', response)))
        plt.savefig('_'.join(('combined', response)))
        plt.close()

        with open('_'.join(('combined', response))+'.txt', 'w') as f:
            f.write('lengths \n')
            f.write(json.dumps(lengths))
            f.write('\n')
            f.write(json.dumps(list(zip(conditions, vals))))

    elif not together:
        for animal in animals:
            vals = []
            for length in lengths:
                response_seq = '_'.join([response]*length)

                # get probabilities
                p = responseSequences([animal], length, esc10=esc10)

                # get successive response values
                try:
                    vals.append(p[response_seq])
                except KeyError:
                    vals.append([0]*3)

            # combine probabilities from each condition
            vals = list(zip(*vals))

            plt.figure()
            for i, val in enumerate(vals):
                plt.plot(lengths, val, label=conditions[i])
            plt.legend()
            plt.xlabel('Number of succesive responses')
            plt.ylabel('Probability of occuring')
            plt.title('_'.join((animal, response)))
            plt.savefig('_'.join((animal, response)))
            plt.close()

            with open('_'.join((animal, response))+'.txt', 'w') as f:
                f.write('lengths \n')
                f.write(json.dumps(lengths))
                f.write('\n')
                f.write(json.dumps(list(zip(conditions, vals))))

    else:
        # plot each condition separately
        for i, condition in enumerate(conditions):
            vals = {a: [] for a in animals}

            for animal in animals:
                for length in lengths:
                    response_seq = '_'.join([response]*length)

                    # get probabilities
                    p = responseSequences([animal], length, esc10=esc10)

                    # get successive response values
                    try:
                        vals[animal].append(p[response_seq][i])
                    except KeyError:
                        vals[animal].append(0)

            fname = '_'.join(('all', condition, response))

            plt.figure()
            for animal in animals:
                plt.plot(lengths, vals[animal], label=animal)
            plt.legend()
            plt.xlabel('Number of succesive responses')
            plt.ylabel('Probability of occuring')
            plt.title(fname)
            plt.savefig(fname)
            plt.close()

            with open(fname+'.txt', 'w') as f:
                f.write('lengths \n')
                f.write(json.dumps(lengths))
                f.write('\n')
                f.write(json.dumps(vals))


def pResponses():
    """save a txt file with the probability of each response for each animal
       during each condition"""

    animals = ['A2', 'A3', 'A7', 'A9', 'A10', 'A11', 'A12']
    with open('pResponses.txt', 'w') as f:
        for a in animals:
            f.write(a)
            f.write(json.dumps(responseSequences([a], 1, esc10=True)))
            f.write('\n')


def seq_viz(length, animal, condition, saveFig=False, showFig=False):
    """plot response sequences - arrow format

    INPUTS:
        length - (int) the sequence length to consider
        animal - (str) animal identifier
        condition - (str) condition of interest
        saveFig - (bool) whether to save the figure
        showFig - (bool) whether to display the figure"""

    responses = ['Premature', 'Failure', 'Escape']
    positions = [str(i) for i in range(2, length + 1, 1)]

    conditions = {'NA': 0, 'LH': 1, 'KET': 2}

    # get data
    seq = responseSequences([animal], length)

    x = [i for i in range(1, length, 1)]
    y = {'Premature': 0.5, 'Failure': 2.5, 'Escape': 4.5}

    arr_color = 'k'

    ax = plt.subplot(111)

    plt.xlim(0, length-0.8)
    plt.ylim(0, 5)

    plt.xlabel('Response position')

    plt.xticks(x, positions)
    plt.yticks(list(y.values()), responses)

    for item in seq.keys():
        # get sequence of responses
        res_sequence = item.split('_')

        # get response pairs
        pairs = [res_sequence[i:i+2] for i in range(len(res_sequence)-1)]

        for position, pair in enumerate(pairs):

            alpha = seq[item][conditions[condition]]

            if position != (len(pairs)-1):
                plt.plot([x[position]-1, x[position]],
                         [y[pair[0]], y[pair[1]]],
                         alpha=alpha, color=arr_color)
            else:
                styles = ('simple, tail_width='+str(5*alpha)
                          + ', head_width=10, head_length=12')

                kw = {'arrowstyle': styles, 'color': arr_color, 'alpha': alpha}
                a = patches.FancyArrowPatch((x[position]-1, y[pair[0]]),
                                            (x[position], y[pair[1]]),
                                            **kw)
                ax.add_patch(a)

    simpleAxis(ax)

    plt.title('_'.join((animal, condition)))

    colors = plt.cm.ScalarMappable(cmap='Greys')
    colors.set_array([0, 1])
    plt.colorbar(colors, ticks=[0, 0.5, 1], shrink=0.5)

    plt.tight_layout()

    if saveFig:
        plt.savefig('_'.join((animal, condition, 'responseSequences')))

        if not showFig:
            plt.close()

    if showFig:
        plt.show()


def get_seq_viz():
    animals = ['B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10']
    conditions = ['NA', 'LH', 'KET']

    for animal in animals:
        for condition in conditions:
            seq_viz(6, animal, condition)


def combination_viz(length, animal, n=10, plot=False, save=False):
    """plot the probability of each combination of a given length

       INPUTS:
           length - (int) the combination length to plot
           animal - (str) the animal identifier
           n - (int) the number of combinations to plot [default=10]
           plot - (bool) whether to plot the data
           save - (bool) whether to save the data to a txt file"""

    conditions = {'NA': 0, 'LH': 1, 'KET': 2}

    # get all response sequences and their probabilities
    seq = responseSequences([animal], length)

    # sort the response sequences based on their summed probability of
    # occurence across all conditions
    sorted_seq = sorted(seq.items(), key=lambda x: sum(x[1]), reverse=True)

    # response sequence names
    seq_names = [sequence[0] for sequence in sorted_seq[:n]]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))

        for condition in conditions.keys():
            # sequence probabilities for the given condition
            seq_vals = [sequence[1][conditions[condition]]
                        for sequence in sorted_seq[:n]]

            plt.plot(seq_names, seq_vals, label=condition)

        simpleAxis(ax)
        plt.title(animal)
        plt.ylim(0, 0.8)
        plt.ylabel('Probability')
        plt.xlabel('Response sequence')

        y = [round(i, 2) for i in np.arange(0, 0.81, 0.2)]
        plt.yticks(y, [str(i) for i in y])
        plt.xticks(rotation=45)

        plt.legend()
        plt.tight_layout()
        plt.savefig('_'.join((animal, str(length),
                    'sequenceCombinations')))
        plt.close(fig)

    if save:
        fname = '_'.join((animal, 'length' + str(length),
                         'sequenceCombinations'))
        with open(fname + '.txt', 'w') as f:
            f.write(json.dumps(sorted_seq[:n]))


def get_combination_viz():
    animals = ['B2', 'B3', 'B4', 'B7', 'B8', 'B9', 'B10']
    for animal in animals:
        for length in range(1, 10, 1):
            combination_viz(length, animal, save=True)


def test_independence():
    """test whether an animal's response at timepoint t is independent of its
       response at timepoint t-1"""

    conditions = ['NA', 'LH', 'KET']
    animals = ['B2', 'B3', 'B4', 'B7', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14']

    for i, condition in enumerate(conditions):
        t_matrix, nPairs = transitionProb(animals, condition, nPairs=True)

        probabilities = responseProb(animals, condition, probability=True)
        rows = t_matrix.index
        columns = list(t_matrix)

        expected = []
        pairs = []

        for row in rows:
            for col in columns:
                expected.append(nPairs * probabilities[row]
                                * probabilities[col])

                pairs.append('_'.join((row, col)))

        chisq = chisquare(f_obs=t_matrix.values.flatten(),
                          f_exp=expected)

        e_minus_o = [round((e - o)/e, 2)
                     for e, o in zip(expected, t_matrix.values.flatten())]

        if i == 0:
            mode = 'w'
        else:
            mode = 'a'

        with open('chisquare_test_independence.txt', mode) as f:
            f.write(condition)
            f.write('\n')
            f.write(json.dumps(list(zip(pairs, e_minus_o))))
            f.write('\n')
            f.write(str(chisq))
            f.write('\n')
            f.write('\n')


def test_independence_marginals():
    """test whether an animal's response at timepoint t is independent of its
       response at timepoint t-1. Use marginals to compute expected
       frequencies"""

    conditions = ['NA', 'LH', 'KET']
    animals = ['B2', 'B3', 'B4', 'B7', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14']

    for i, condition in enumerate(conditions):
        t_matrix, nPairs = transitionProb(animals, condition, nPairs=True)

        counts = responseProb(animals, condition, counts=True)

        rows = t_matrix.index
        columns = list(t_matrix)

        expected = []
        pairs = []

        for row in rows:
            for col in columns:
                expected.append((counts[row] * counts[col]) / nPairs)
                pairs.append('_'.join((row, col)))

        chisq = chisquare(f_obs=t_matrix.values.flatten(),
                          f_exp=expected)

        e_minus_o = [round((e - o)/e, 2)
                     for e, o in zip(expected, t_matrix.values.flatten())]

        if i == 0:
            mode = 'w'
        else:
            mode = 'a'

        with open('chisquare_test_independence_marginals.txt', mode) as f:
            f.write(condition)
            f.write('\n')
            f.write(json.dumps(list(zip(pairs, e_minus_o))))
            f.write('\n')
            f.write(str(chisq))
            f.write('\n')
            f.write('\n')
