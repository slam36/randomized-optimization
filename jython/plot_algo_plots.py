import numpy as np
import matplotlib.pyplot as plt
import csv

algos = ['four_peaks', 'knapsack', 'travelingsalesman']
algos = ['knapsack']
plotcount = 1
for algo in algos:
    key_names = ['rhc_results', 'rhc_times', 'sa_results', 'sa_times', 'ga_results', 'ga_times', 'mimic_results', 'mimic_times']
    data = dict.fromkeys(key_names)
    with open(algo + '.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        count = 0
        for row in csvreader:
            data[key_names[count]] = row
            count += 1

    iters = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    data_avg = dict.fromkeys(key_names)
    data_std = dict.fromkeys(key_names)

    for i in key_names:
        currentRow = data[i]
        composite_list = [currentRow[x:x+5] for x in range(0, len(currentRow), 5)]
        current_avgs = []
        current_stds = []
        for j in composite_list:
            j = list(map(float, j))
            current_avgs.append(np.mean(j))
            current_stds.append(np.std(j))
        data_avg[i] = current_avgs
        data_std[i] = current_stds


    print(iters)
    print(data_avg['rhc_results'])
    plt.figure(plotcount)
    plt.plot(iters, data_avg['rhc_results'], marker = 'o', linewidth=2, color='r', label='RHC')
    plt.plot(iters, data_avg['sa_results'], marker = 'o', linewidth=2, color='g', label='SA')
    plt.plot(iters, data_avg['ga_results'], marker = 'o', linewidth=2, color='b', label='GA')
    plt.plot(iters[0:6], data_avg['mimic_results'], marker = 'o', linewidth=2, color='k', label='MIMIC')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.xlim(0, 2500)
    plt.legend()
    plt.savefig(algo + '_fitness.png')
    plotcount += 1

    plt.figure(plotcount)
    plt.plot(iters, data_std['rhc_results'], marker = 'o', linewidth=2, color='r', label='RHC')
    plt.plot(iters, data_std['sa_results'], marker = 'o', linewidth=2, color='g', label='SA')
    plt.plot(iters, data_std['ga_results'], marker = 'o', linewidth=2, color='b', label='GA')
    plt.plot(iters[0:6], data_std['mimic_results'], marker = 'o', linewidth=2, color='k', label='MIMIC')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Standard Deviation')
    plt.legend()
    plt.xlim(0, 2500)
    plt.savefig(algo + '_stdev.png')
    plotcount += 1

    plt.figure(plotcount)
    plt.plot(iters, data_avg['rhc_times'], marker = 'o', linewidth=2, color='r', label='RHC')
    plt.plot(iters, data_avg['sa_times'], marker = 'o', linewidth=2, color='g', label='SA')
    plt.plot(iters, data_avg['ga_times'], marker = 'o', linewidth=2, color='b', label='GA')
    plt.plot(iters[0:6], data_avg['mimic_times'], marker = 'o', linewidth=2, color='k', label='MIMIC')
    plt.xlabel('Iterations')
    plt.ylabel('Wallclock Time (s)')
    plt.legend()
    plt.xlim(0, 2500)
    plt.savefig(algo + '_times.png')
    plotcount += 1





































