"""
This file uses the evolutionary algorithm framework to sovle Sudoku puzzles
By Madelyn Redick
"""
import numpy as np
import operator as op
from evolutionary_framework import Evo
import random as rnd
import pandas as pd
# from profiler import Profiler TODO
import csv
import multiprocessing
import time as Time
from functools import reduce

""" TODO
def profile(f):
    return Profiler.profile(f)
    
we used a profiler to initially time our functions but multiprocessing does not allow for profiling so we have 
commented it out
"""


# @profile TODO
def max_pref(row_will, max):
    try:
        return [1 if j in rnd.sample([i for i in range(len(row_will)) if row_will[i] == 'P' or row_will[i] ==
                                      'W'], max) else 0 for j in range(len(row_will))]
    except ValueError:
        return [0] * len(max)


# @profile TODO
def pd_numpy(df, col1, col2):
    """Takes certain columns in dataframe and turns them to 2d numpy array"""
    return df[list(df)[col1:col2 + 1]].to_numpy()


# @profile TODO
def conflict_time(lst):
    """
    Takes a zipped list and if the length of the set is different that the length of the list then there were repeats
    """

    return 0 if len((set([(w, t) for w, t in lst if w == 1]))) == len([(w, t) for w, t in lst if w == 1]) else 1


# create objectives
# @profile TODO
def overallocation(L, max):
    """Sums the total overallocation of TAs

    args:
        max (list): a list of TAS max assign
        L: a 2d array of current assignments with each row representing a TA

    returns total overallocation"""
    return sum([sum(L[i]) - max[i] for i in range(len(max)) if sum(L[i]) - max[i] > 0])


# @profile TODO
def conflicts_np(row, time):
    return 0 if len((set([(w, t) for w, t in list(zip(row, time))
                          if w == 1]))) == len([(w, t) for w, t in list(zip(row, time)) if w == 1]) else 1


# @profile TODO
def conflicts(L, time):
    """Sums the total number of conflicts (multiple conflicts of same TA counts as 1

    args:
        time: list of times the labs
        L: 2d array of current assignments with each row representing a TA

    returns total conflicts"""

    # return sum(np.apply_along_axis(lambda x: conflicts_np(x, time), axis=1, arr=L))
    return sum([0 if len((set([(w, t) for w, t in assign if w == 1]))) == len([(w, t) for w, t in assign if w == 1])
                else 1 for assign in [list(zip(row, time)) for row in L]])


# @profile TODO
def under_support(L, min):
    """
    Sums the total amount of undersupport through all lab sections

    args:
        L: 2d numpy array with each row representing a TA
        min: a list of minimum lab requirements

    returns: total undersupport
    """
    return sum([min[i] - sum(L.T[i]) for i in range(len(min)) if min[i] - sum(L.T[i]) > 0])


# @profile TODO
def unwilling(L, W):
    """
    sums the total number of times you assign a TA to a section they are unwilling to teach

    args:
        L: 2d numpy array with each row representing a TAs assignments
        W: 2d numpy array with each row representing a TAs willingness

    returns:
        total unwilling assignments
    """
    return sum([op.countOf(zip(W[i], L[i]), ('U', 1)) for i in range(len(L))])


# @profile TODO
def unpreffered(L, W):
    """
    sums the total number of times you assign a TA to a section they are unwilling to teach

    args:
        L: 2d numpy array with each row representing a TAs assignments
        W: 2d numpy array with each row representing a TAs willingness

    returns:
        total willing but not preferred assignments
    """
    return sum([op.countOf(zip(W[i], L[i]), ('W', 1)) for i in range(len(L))])


# @profile TODO
def max_p(_, W, max):
    """randomly assign tas to sections they are willing to teach
    This will be my starter solution"""

    """
    return np.array([[1 if k in rnd.sample([j for j in range(len(W[i])) if W[i][j] == 'P' or W[i][j] ==
                                            'W'], max[i]) else 0 for k in range(len(W[i]))]
                     if 'P' in W[i] or 'W' in W[i] else [0] * len(W[i]) for i in range(len(W))])
                     
    attempt to put same logic into list comprehension 
    """

    return np.array([max_pref(W[i], max[i]) for i in range(len(W))])


# @profile TODO
def flip(row, max):
    """takes a row and randomly flips 0, 1 values based on max"""
    return np.array([(row[i] + 1) % 2 if i in rnd.sample(range(len(row)), max) else row[i] for i in range(len(row))])


# @profile TODO
def swap(L, max):
    """change somewhere between 0 and max"""
    L = np.array([flip(L[i], max[i]) for i in range(len(L))])
    return L


# @profile TODO
def swap_1(L):
    """Flip a random value in a random of row"""
    i, j = rnd.choice(range(len(L))), rnd.choice(range(len(L.T)))
    L[i], L[j] = (L[i] + 1) % 2, (L[j] + 1) % 2
    return L


def evolve(nds, E):
    """Calls the evolve function from the evolutionary framework (for some reason was not allowing us to do this
    directly from the multiprocessors"""
    E.evolve(nds, 5)
    return nds


class AllNumsClass:
    """Credit to "https://stackoverflow.com/questions/55643339/python-multiprocessing-sharing-data-between-processes"
 for demonstrating this sharing process"""

    def __init__(self):
        self.queue = multiprocessing.Queue()

    def get_queue(self):
        return self.queue


def save_solutions_to_csv(nds, filename):
    """
    saves non dominated set to csv
    nds: dictionary of non dominated set
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['groupname', 'overallocation', 'conflicts', 'under', 'unwilling', 'unpreffered']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        groupname = 'samadje'

        for objectives, sol in nds.items():
            row = {'groupname': groupname}
            row.update(dict(objectives))
            writer.writerow(row)


def mp(process, method, nds, start, length, queue):
    """
    process (int) number of processes to run
    method (function): function to run
    nds (dict): non dominated set
    start (time object): time function starts
    length (int): seconds function runs for
    queue (object): class that allows for multiprocessing queues and shared data

    returns: new non dominated set
    """
    while Time.time() - start < length:
        processes = []
        for p in range(process):
            processes.append(multiprocessing.Process(target=evolve, args=(nds, method)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        while not queue.get_queue().empty():
            nds.update(queue.get_queue().get())

    return nds


def dominates(p, q):
    """returns whether p dominates q"""
    pscores = [score for _, score in p]
    qscores = [score for _, score in q]
    score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
    min_diff = min(score_diffs)
    max_diff = max(score_diffs)
    return min_diff >= 0.0 and max_diff > 0.0


def reduce_nds(S, p):
    """finds non dominated set"""
    S = set(S)
    return S - {q for q in S if dominates(p, q)}


def remove_dominated(nds):
    """create new non_dominated set"""
    ndsname = reduce(reduce_nds, nds.keys(), nds.keys())
    nds = {k: nds[k] for k in ndsname}
    return nds

def main():
    # load in data
    tas = pd.read_csv('sudoku_500.csv') #previouslt called tas
    sections = pd.read_csv('sections.csv') #previouslt called sections
    #test = pd.read_csv('test1.csv', header=None) #TODO

    # find needed objective arrays (the arrays our objective functions need to calculate)
    W = pd_numpy(tas, 3, 20)
    min = sections['min_ta'].to_numpy()
    time = sections['daytime'].to_numpy()
    max = tas['max_assigned'].to_numpy()

    # Creating an instance of the framework
    E = Evo()  # tas, sections)

    # Register all objectives (fitness criteria)
    E.add_fitness_criteria('overallocation', overallocation, max)
    E.add_fitness_criteria('conflicts', conflicts, time)
    E.add_fitness_criteria('under', under_support, min)
    E.add_fitness_criteria('unwilling', unwilling, W)
    E.add_fitness_criteria('unpreffered', unpreffered, W)

    # add solution and confirm test case
    #E.add_solution(test.to_numpy())

    # add agents
    E.add_agent("max_p", max_p, W, max, k=1)
    E.add_agent("swap", swap, max, k=1)
    E.add_agent('swap1', swap_1, k=1)

    # initialize the starting nds dictionary, the length of the runtime and the queueing class
    nds = multiprocessing.Manager().dict()
    length, start = 600, Time.time()
    all_nums_class = AllNumsClass()

    # multiprocess and get the new non dominated set
    nds = mp(4, E, nds, start, length, all_nums_class)
    nds = remove_dominated(nds)
    print(nds) # use this printout to find a solution of interest

    # save non-dominated solutions as csv
    save_solutions_to_csv(nds, 'nondominated_solutions.csv')

    # Profiler.report() TODO


if __name__ == "__main__":
    main()