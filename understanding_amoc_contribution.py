import matplotlib.pyplot as plt
import numpy as np
import glob

file = '/home/ph290/box_modelling/boxmodel_6_box_back_to_basics_tuning/resultsb_amoc/spg_box_model_qump_results_1_512.csv'

input1 = np.genfromtxt(file, delimiter=",")

for i in range(26):
	y = input1[:,1+i] - np.mean(input1[0:20,1+i])
	plt.plot(input1[:,0],((y/1.12e13)*1.0e15/12.0))

plt.show()


