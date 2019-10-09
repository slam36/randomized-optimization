import numpy as np
import matplotlib.pyplot as plt
import csv



key_names = ['15', '15time', '35', '35time', '55', '55time', '75', '75time', '95', '95time']
data = dict.fromkeys(key_names)
with open('four_peaks_exploringSA.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    count = 0
    for row in csvreader:
        data[key_names[count]] = row
        count += 1

iters = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 30000, 35000, 40000, 45000, 50000]
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


plt.figure(1)
cooling_nums = [15, 35, 55, 75, 95]
for i in cooling_nums:
    plt.plot(iters, data_avg[str(i)], marker='o', linewidth=2, label='Cooling = 0.' + str(i))


plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('exploringSA_fitness.png')







































