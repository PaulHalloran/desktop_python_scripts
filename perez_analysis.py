import numpy as np
import matplotlib.pyplot as plt

in_file1='/home/ph290/data0/box_model_obs/results/box_model_qump_results_stg.csv'
in_file2='/home/ph290/data0/box_model_obs/results/box_model_qump_results_spg.csv'

input1=np.genfromtxt(in_file1, delimiter=",")
input2=np.genfromtxt(in_file2, delimiter=",")

fig = plt.figure()
ax1=plt.subplot2grid((1,2), (0, 0))
for i in np.arange(1000):
    ax1.plot(input1[:,0],input1[:,1+i])
ax2=plt.subplot2grid((1,2), (0, 1))
for i in np.arange(1000):
    ax2.plot(input2[:,0],input2[:,1+i])

plt.show()

