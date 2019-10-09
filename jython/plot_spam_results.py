import numpy as np
import matplotlib.pyplot as plt
import csv

train_error = 0
test_error = 1
train_acc = 2
test_acc = 3

rhc_file = 'RHC_spam_errors.csv'
rhc = np.genfromtxt(rhc_file, delimiter=',')
numDataPts = len(rhc)
rhc = np.transpose(rhc)
iter = range(5000)
plt.figure(1)
#normalize by number of data points so that the values somewhat mean something
plt.plot(iter, rhc[:][train_error] / 0.7 / numDataPts, label='Training') 
plt.plot(iter, rhc[:][test_error] / 0.3 / numDataPts, label='Testing')
plt.xlabel('Iterations')
plt.ylabel('Least Squares Error / (Training/Testing) Size')
plt.legend()
plt.savefig('rhc_spam_errors.png')

plt.figure(2)
plt.plot(iter, rhc[:][train_acc], label='Training')
plt.plot(iter, rhc[:][test_acc], label='Testing')
plt.xlabel('Iterations')
plt.ylabel('Accuracy %')
plt.legend()
plt.savefig('rhc_spam_acc.png')

cooling = [15, 35, 55, 75, 95]
things_to_plot = [train_error, test_error, train_acc, test_acc]
SA_files = ['SA_' + str(x) + '_spam_errors.csv' for x in cooling]
y_labels = ['Least Squares Error / Training Size', 'Least Squares Error / Testing Size', 'Accuracy %', 'Accuracy %']
output_files = ['SA_training_error.png', 'SA_testing_error.png', 'SA_training_acc.png', 'SA_testing_acc.png']
figure_count = 3
for thing in range(len(things_to_plot)):
    plt.figure(figure_count)
    for f in range(len(SA_files)):
        sa = np.genfromtxt(SA_files[f], delimiter=',')
        sa = np.transpose(sa)
        if thing == 0:
            plt.plot(iter, sa[:][thing] / 0.7 / numDataPts, label=('Cooling = 0.' + str(cooling[f])))
        elif thing == 1:
            plt.plot(iter, sa[:][thing] / 0.3 / numDataPts, label=('Cooling = 0.' + str(cooling[f])))
        else:
            plt.plot(iter, sa[:][thing], label=('Cooling = 0.' + str(cooling[f])))
        
    plt.xlabel('Iterations')
    plt.ylabel(y_labels[thing])
    plt.legend()
    plt.savefig(output_files[thing])
    figure_count += 1

iter = range(500)
parameters = [(100, 50, 5), (100, 50, 10), (100, 100, 5), (100, 100, 10), (200, 50, 5), (200, 50, 10), (200, 100, 5), (200, 100, 10)]
things_to_plot = [train_error, test_error, train_acc, test_acc]
GA_files = ['GA_' + str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]) + '_spam_errors.csv' for x in parameters]
y_labels = ['Least Squares Error / Training Size', 'Least Squares Error / Testing Size', 'Accuracy %', 'Accuracy %']
output_files = ['GA_training_error.png', 'GA_testing_error.png', 'GA_training_acc.png', 'GA_testing_acc.png']
for thing in range(len(things_to_plot)):
    plt.figure(figure_count)
    for f in range(len(GA_files)):
        ga = np.genfromtxt(GA_files[f], delimiter=',')
        ga = np.transpose(ga)
        if thing == 0:
            plt.plot(iter, ga[:][thing] / 0.7 / numDataPts, label=(str(parameters[f])))
        elif thing == 1:
            plt.plot(iter, ga[:][thing] / 0.3 / numDataPts, label=(str(parameters[f])))
        else:
            plt.plot(iter, ga[:][thing], label=(str(parameters[f])))

    plt.xlabel('Iterations')
    plt.ylabel(y_labels[thing])
    plt.legend()
    plt.savefig(output_files[thing])
    figure_count += 1



    
    
















