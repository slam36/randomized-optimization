import numpy as np
import matplotlib.pyplot as plt


parameters = ['(100, 50, 5)', '(100, 50, 10)', '(100, 100, 5)', '(100, 100, 10)', '(200, 50, 5)', '(200, 50, 10)', '(200, 100, 5)', '(200, 100, 10)']
acc = [88.6, 91.6, 90.4, 89.5, 90.4, 86.6, 90.8, 90.7]
plt.figure(1)
plt.plot(parameters, acc, marker='o')
plt.xticks(rotation=45, ha="right")
plt.xlabel('(population size, number to mate / iteration, number to mutate / iteration')
plt.ylabel('Accuracy')
plt.show()
#plt.savefig('plot_spam_ga_bar.png')


