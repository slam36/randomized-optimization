import sys
import os
import time

sys.path.append("../ABAGAIL.jar")

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array

import csv




"""
Commandline parameter(s):
    none
"""

'''
RHC: 3136.13731444
SA: 3196.36904655
GA: 3603.19902689
MIMIC: 3787.99974185
'''

def run_knapsack():
    # Random number generator */
    random = Random()
    # The number of items
    NUM_ITEMS = 40
    # The number of copies each
    COPIES_EACH = 4
    # The maximum weight for a single element
    MAX_WEIGHT = 50
    # The maximum volume for a single element
    MAX_VOLUME = 50
    # The volume of the knapsack 
    KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

    # create copies
    fill = [COPIES_EACH] * NUM_ITEMS
    copies = array('i', fill)

    # create weights and volumes
    fill = [0] * NUM_ITEMS
    weights = array('d', fill)
    volumes = array('d', fill)
    for i in range(0, NUM_ITEMS):
        weights[i] = random.nextDouble() * MAX_WEIGHT
        volumes[i] = random.nextDouble() * MAX_VOLUME


    # create range
    fill = [COPIES_EACH + 1] * NUM_ITEMS
    ranges = array('i', fill)



    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = UniformCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

    iters = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    num_repeats = 5

    rhc_results = []
    rhc_times = []
    for i in iters:
        print(i)
        for j in range(num_repeats):
            start = time.time()
            rhc = RandomizedHillClimbing(hcp)
            fit = FixedIterationTrainer(rhc, i)
            fit.train()
            end = time.time()
            rhc_results.append(ef.value(rhc.getOptimal()))
            rhc_times.append(end - start)
            #print "RHC: " + str(ef.value(rhc.getOptimal()))


    sa_results = []
    sa_times = []
    for i in iters:
        print(i)
        for j in range(num_repeats):
            start = time.time()
            sa = SimulatedAnnealing(100, .95, hcp)
            fit = FixedIterationTrainer(sa, i)
            fit.train()
            end = time.time()
            
            sa_results.append(ef.value(sa.getOptimal()))
            sa_times.append(end - start)
            #print "SA: " + str(ef.value(sa.getOptimal()))


    ga_results = []
    ga_times = []
    for i in iters:
        print(i)
        for j in range(num_repeats):
            start = time.time()
            ga = StandardGeneticAlgorithm(200, 150, 25, gap)
            fit = FixedIterationTrainer(ga, i)
            fit.train()
            end = time.time()
            ga_results.append(ef.value(sa.getOptimal()))
            ga_times.append(end - start)
            #print "GA: " + str(ef.value(ga.getOptimal()))

    mimic_results = []
    mimic_times = []
    for i in iters[0:6]:
        print(i)
        for j in range(num_repeats):
            start = time.time()
            mimic = MIMIC(200, 100, pop)
            fit = FixedIterationTrainer(mimic, i)
            fit.train()
            end = time.time()
            mimic_results.append(ef.value(mimic.getOptimal()))
            mimic_times.append(end - start)
            #print "MIMIC: " + str(ef.value(mimic.getOptimal()))

    with open('knapsack.csv', 'w') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(rhc_results)
      writer.writerow(rhc_times)
      writer.writerow(sa_results)
      writer.writerow(sa_times)
      writer.writerow(ga_results)
      writer.writerow(ga_times)
      writer.writerow(mimic_results)
      writer.writerow(mimic_times)

    return rhc_results, rhc_times, sa_results, sa_times, ga_results, ga_times, mimic_results, mimic_times

if __name__ == "__main__":
   run_knapsack()