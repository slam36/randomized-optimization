import os
import sys
import csv
import time
import random

sys.path.append("../ABAGAIL.jar")

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import ReLU

#from __future__ import with_statement

# This function is adapted from the ABAGAIL provided abalone_test.py

x_train_file = 'x_train.csv'
x_test_file = 'x_test.csv'
y_train_file = 'y_train.csv'
y_test_file = 'y_test.csv'

INPUT_LAYER = 57
HIDDEN_LAYER = 57
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5000

def initialize_instances():
    x_train = []
    with open(x_train_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            x_train.append(row)

    x_test = []
    with open(x_test_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            x_test.append(row)

    y_train = []
    with open(y_train_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            y_train.append(row)
    
    y_test = []
    with open(y_test_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            y_test.append(row)

    trainingInstances = []
    testingInstances = []
    for i in range(len(x_train)):
        instance = Instance([float(value) for value in x_train[i]])
        instance.setLabel(Instance(float(y_train[i][0])))
        trainingInstances.append(instance)

    for i in range(len(x_test)):
        instance = Instance([float(value) for value in x_test[i]])
        instance.setLabel(Instance(float(y_test[i][0])))
        testingInstances.append(instance)

    
    return trainingInstances, testingInstances

def train(oa, network, oaName, trainingInstances, testingInstances, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] trainingInstances
    :param list[Instance] testingInstances
    :param AbstractErrorMeasure measure:
    """


    print "\nError results for %s\n---------------------------" % (oaName,)
    with open(oaName + '_spam_errors.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        for iteration in xrange(TRAINING_ITERATIONS):
            oa.train()
            train_error = 0.00
            test_error = 0.00
            traincorrect = 0
            trainincorrect = 0
            testcorrect = 0
            testincorrect = 0

            for instance in trainingInstances:
                network.setInputValues(instance.getData())
                network.run()

                output = instance.getLabel()
                output_values = network.getOutputValues()
                example = Instance(output_values, Instance(output_values.get(0)))
                train_error += measure.value(output, example)

                predicted = instance.getLabel().getContinuous()
                actual = network.getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    traincorrect += 1
                else:
                    trainincorrect += 1

            for instance in testingInstances:
                network.setInputValues(instance.getData())
                network.run()

                output = instance.getLabel()
                output_values = network.getOutputValues()
                example = Instance(output_values, Instance(output_values.get(0)))
                test_error += measure.value(output, example)

                predicted = instance.getLabel().getContinuous()
                actual = network.getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    testcorrect += 1
                else:
                    testincorrect += 1

            train_acc = float(traincorrect)/(traincorrect+trainincorrect)*100.0
            test_acc = float(testcorrect)/(testcorrect+testincorrect)*100.0

            print(oaName + " iter = " + str(iteration) + " train error: " + str(train_error) + ", test error: " + str(test_error) + ", train accuracy: " + str(train_acc) + ", test accuracy: " + str(test_acc))
            writer.writerow([float(train_error), float(test_error), float(train_acc), float(test_acc)])

        
def main():
    trainingInstances, testingInstances = initialize_instances()
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()

    data_set = DataSet(trainingInstances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA_15", "SA_35", "SA_55", "SA_75", "SA_95"]
    #oa_names=["GA_100_50_5", "GA_200_50_5", "GA_100_50_10", "GA_200_50_10", "GA_100_100_5", "GA_200_100_5", "GA_100_100_10", "GA_200_100_10"]
    #oa_names=["GA_200_100_5", "GA_100_100_10", "GA_200_100_10"]
    
    for name in oa_names:
        #use RELU activation function
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER], ReLU())
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    
    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .15, nnop[1]))
    oa.append(SimulatedAnnealing(1E11, .35, nnop[2]))
    oa.append(SimulatedAnnealing(1E11, .55, nnop[3]))
    oa.append(SimulatedAnnealing(1E11, .75, nnop[4]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[5]))

    # oa.append(StandardGeneticAlgorithm(100, 50, 5, nnop[0]))
    # oa.append(StandardGeneticAlgorithm(200, 50, 5, nnop[1]))
    # oa.append(StandardGeneticAlgorithm(100, 50, 10, nnop[2]))
    # oa.append(StandardGeneticAlgorithm(200, 50, 10, nnop[3]))
    # oa.append(StandardGeneticAlgorithm(100, 100, 5, nnop[4]))
    #oa.append(StandardGeneticAlgorithm(200, 100, 5, nnop[0]))
    #oa.append(StandardGeneticAlgorithm(100, 100, 10, nnop[1]))
    #oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    with open('nn_spam_results_RHC_SA.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i, name in enumerate(oa_names):
            results = ''

            start = time.time()
            traincorrect = 0
            trainincorrect = 0
            testcorrect = 0
            testincorrect = 0

            train(oa[i], networks[i], oa_names[i], trainingInstances, testingInstances, measure)
            end = time.time()
            training_time = end - start

            optimal_instance = oa[i].getOptimal()
            networks[i].setWeights(optimal_instance.getData())

            start = time.time()
            for instance in trainingInstances:
                networks[i].setInputValues(instance.getData())
                networks[i].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    traincorrect += 1
                else:
                    trainincorrect += 1

            for instance in testingInstances:
                networks[i].setInputValues(instance.getData())
                networks[i].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    testcorrect += 1
                else:
                    testincorrect += 1

            end = time.time()
            testing_time = end - start

            results += "\nResults for %s: \nCorrectly classified %d training instances." % (name, traincorrect)
            results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (trainincorrect, float(traincorrect)/(traincorrect+trainincorrect)*100.0)
            results += "\nResults for %s: \nCorrectly classified %d testing instances." % (name, testcorrect)
            results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (testincorrect, float(testcorrect)/(testcorrect+testincorrect)*100.0)
            results += "\nTraining time: %0.03f seconds" % (training_time,)
            results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

            print results
            writer.writerow([results])
            writer.writerow('')
            
    


if __name__ == "__main__":
    main()
























