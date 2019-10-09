# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
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
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
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
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array

import csv


"""
Commandline parameter(s):
    none
"""

'''
RHC Inverse of Distance: 0.121112899393
SA Inverse of Distance: 0.130402005563
GA Inverse of Distance: 0.149654726999
MIMIC Inverse of Distance: 0.112714069889
'''

def run_traveling_salesman():
    # set N value.  This is the number of points
    N = 50
    random = Random()

    points = [[0 for x in xrange(2)] for x in xrange(N)]
    for i in range(0, len(points)):
        points[i][0] = random.nextDouble()
        points[i][1] = random.nextDouble()

    ef = TravelingSalesmanRouteEvaluationFunction(points)
    odd = DiscretePermutationDistribution(N)
    nf = SwapNeighbor()
    mf = SwapMutation()
    cf = TravelingSalesmanCrossOver(ef)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

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
            print "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))
            # print "Route:"
            # path = []
            # for x in range(0,N):
            #     path.append(rhc.getOptimal().getDiscrete(x))
            # print path

    sa_results = []
    sa_times = []
    for i in iters:
        print(i)
        for j in range(num_repeats):
            start = time.time()
            sa = SimulatedAnnealing(1E12, .999, hcp)
            fit = FixedIterationTrainer(sa, i)
            fit.train()
            sa_results.append(ef.value(sa.getOptimal()))
            sa_times.append(end - start)
            print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))
            # print "Route:"
            # path = []
            # for x in range(0,N):
            #     path.append(sa.getOptimal().getDiscrete(x))
            # print path

    ga_results = []
    ga_times = []
    for i in iters:
        print(i)
        for j in range(num_repeats):
            start = time.time()
            ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
            fit = FixedIterationTrainer(ga, i)
            fit.train()
            end = time.time()
            ga_results.append(ef.value(ga.getOptimal()))
            print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))
            ga_times.append(end - start)
            # print "Route:"
            # path = []
            # for x in range(0,N):
            #     path.append(ga.getOptimal().getDiscrete(x))
            # print path


    # for mimic we use a sort encoding
    ef = TravelingSalesmanSortEvaluationFunction(points)
    fill = [N] * N
    ranges = array('i', fill)
    odd = DiscreteUniformDistribution(ranges)
    df = DiscreteDependencyTree(.1, ranges); 
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

    mimic_results = []
    mimic_times = []
    for i in iters[0:6]:
        print(i)
        for j in range(num_repeats):
            start = time.time()
            mimic = MIMIC(500, 100, pop)
            fit = FixedIterationTrainer(mimic, i)
            fit.train()
            end = time.time()
            
            mimic_results.append(ef.value(mimic.getOptimal()))
            print "MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal()))
            # print "Route:"
            # path = []
            # optimal = mimic.getOptimal()
            # fill = [0] * optimal.size()
            # ddata = array('d', fill)
            # for i in range(0,len(ddata)):
            #     ddata[i] = optimal.getContinuous(i)
            # order = ABAGAILArrays.indices(optimal.size())
            # ABAGAILArrays.quicksort(ddata, order)
            # print order
            mimic_times.append(end - start)

    with open('travelingsalesman.csv', 'w') as csvfile:
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
    run_traveling_salesman()
