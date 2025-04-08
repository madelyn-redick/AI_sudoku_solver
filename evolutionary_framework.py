""" 
Evolutionary algorithm framework for solving Sudoku puzzles
By Madelyn Redick
"""
import copy
import random as rnd
from functools import reduce
import time

class Evo:

    def __init__(self):  # , tas, section):
        self.pop = {}  # eval -> solution   eval = ((name1, val1), (name2, val2)..)
        self.fitness = {}  # name -> function
        self.agents = {}  # name -> (operator, num_solutions_input)

    def add_fitness_criteria(self, name, f, *args):
        self.fitness[name] = (f, *args)

    def add_agent(self, name, op, *args, k=1):
        self.agents[name] = (op, *args, k)

    def add_solution(self, sol):
        eval = tuple([(name, f[0](sol, f[1])) for name, f in self.fitness.items()])
        self.pop[eval] = sol

    def get_random_solutions(self, k=1):
        popvals = tuple(self.pop.values())
        return [copy.deepcopy(rnd.choice(popvals)) for _ in range(k)]

    def run_agent(self, name):
        op, *args, k = self.agents[name]

        # returns a list based on number of solutions we want to try
        picks = self.get_random_solutions(k)
        if k == 1:
            new_solution = op(picks[0], *args)
        self.add_solution(new_solution)

    @staticmethod
    def _dominates(p, q):
        """returns whether p dominates q"""
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Evo._dominates(p, q)}

    def remove_dominated(self):
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}

    def evolve(self, nds, n=30, dom=250):
        """takes a current non dominated set and runs agents for n (int) time computing the non dominated set every
        250 iterations"""
        # initialize current nds
        self.pop.update(nds)

        # run for desired length with a variety of agents
        length, start = n, time.time()
        sols = 0
        while time.time() - start < length/10:
            self.run_agent('max_p')
            sols+=1
            if sols % dom==0:
                self.remove_dominated()
        while time.time() - start < length/5:
            self.run_agent('swap')
            sols+=1
            if sols % dom==0:
                self.remove_dominated()
        while time.time() - start < length:
            self.run_agent('swap1')
            sols+=1
            if sols % dom==0:
                self.remove_dominated()

        # test to see if multiprocessing works
        #print(sols)

        # find new nds and update current nds
        self.remove_dominated()
        nds.update(self.pop)
        return nds

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + '\n'
        return rslt

# run the agent that finds the max pref as long as we can and then once it stops being as effective go to a more
# random approach