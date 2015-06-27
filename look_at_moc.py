import numpy as np
import glob
import matplotlib.pyplot as plt
import time

directory1 = '/home/ph290/data1/qump_out_python/annual_means/'

directory2 = '/home/ph290/data1/qump_out_python/annual_means/band_pass/'

files1 = glob.glob(directory1+'qump_data_run_*_moc_stm_fun.txt')
files2 = glob.glob(directory2+'qump_data_run_*_moc_stm_fun.txt')

for i in range(3):
	plt.close('all')
	data1 = np.genfromtxt(files1[i],delimiter = ',')
	data2 = np.genfromtxt(files2[i],delimiter = ',')
	plt.plot(data1[:,0],data1[:,1],'r')
	plt.plot(data1[:,0],data2[:,1],'b')
	plt.show(block = True)