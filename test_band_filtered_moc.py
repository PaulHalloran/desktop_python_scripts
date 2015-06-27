import numpy as np
import matplotlib.pyplot as plt

file='qump_data_run_akiqj_stash_200.txt'
file1='/home/ph290/data1/qump_out_python/annual_means/'+file
file2='/home/ph290/data1/qump_out_python/annual_means/band_pass_2/'+file

input1=np.genfromtxt(file1, delimiter=",")
input1=input1[:,0:2]
input2=np.genfromtxt(file2, delimiter=",")
input2=input2[:,0:2]

plt.plot(input1[:,0],input1[:,1],'b')
plt.plot(input2[:,0],input2[:,1],'r')
plt.show()


params=np.genfromtxt('/home/ph290/data1/data/piston_hypercube_7param_1000exp_wide_2013b.csv', delimiter=",")
