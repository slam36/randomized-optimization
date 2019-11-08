# Exploration of Randomized Optimization algorithms
In this project, I explore random search optimization algorithms. These algorithms were randomized hill climbing, simulated annealing, genetic algorithm, and mutual information maximizing input clustering (MIMIC). To explore the strengths and weaknesses of each algorithm, each algorithm was applied to three common optimizatoin problems: four peaks, knapsack, and traveling salesman. In addition to analyzing discrete optimization problems, these algorithms were also used to find optimal weights for a neural network in place of back propagation. I used the ABAGAIL library which is implemented in Java but I used jython for this project.

### Note: This code is a copy of the ABAGAIL code from: https://github.com/pushkar/ABAGAIL 
### What I did was take the starter code from the jython directory and edit it for this project.

## Instructions to run code

Before running, make sure you do
$ant
in the main directory to generate ABAGAIL.jar 
ABAGAIL.jar must be in the main directory for all the code to work because the jython files search relatively for this file.

In the jython directory, the code files that has changed / were created for this project are:
(Note that some files are for running optimizations, and some are for plotting. The optimization ones need to be run with jython while the plotting ones need to be run with python because jython does not have numpy and matplotlib. Run all the jython files first, and then the python files. The jython files all produce .csv files which the python files read to plot.)

### fourpeaks.py (jython) 
(Modified from starter code - run algos on fourpeaks and produce .csv files of results)
### fourpeaks_exploringSA.py (jython) 
(Modified from fourpeaks.py - run SA with different cooling rates on fourpeaks)
### knapsack.py (jython) 
(Modified from starter code - run algos on knapsack and produce .csv files of results)
### plot_algo_plots.py (python) 
(Plot results from fourpeaks, knapsack, and traveling salesman using the .csv files produced)
### plot_exploringSA.py (python) 
(Plot results from fourpeaks_exploringSA.py)
### plot_spam_ga_bar.py (python) 
(This was not actually used for the report)
### plot_spam_results.py (python) 
(Plot all the neural network optimization results)
### spam_nn_GA.py (jython) 
(Modified from abalone-test.py starter code - run genetic algorithm on spambase dataset and produce .csv files to plot)
### spam_nn_RHC_SA.py (jython)
(Modified from abalone-test.py starter code - run rhc and sa algorithm on spambase dataset and produce .csv files to plot)
### travelingsalesman.py (jython) 
(Modified from starter code - run algos on travelingsalesman problem and produce .csv files of results)


The data files for the neural network section are:
x_test.csv, x_train.csv, y_test.csv, and y_train.csv


Note that all the code has already been run, so the output .csv and .png files are already produced. You can always delete them and rerun it, or just run the code and they will be overwritten. 









