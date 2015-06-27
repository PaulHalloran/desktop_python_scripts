import glob
import numpy as np
import random
import subprocess

'''
DONT RUN AGAIN!

file_names = glob.glob('/home/ph290/data1/qump_out_python/annual_means/*.txt')

run_names = []

for file_name in file_names:
	run_names.append(file_name.split('/')[-1].split('_')[3])

run_names = list(np.unique(run_names))
for i in range(int(np.floor(np.size(run_names)/2.0))):
	run = random.choice(run_names)
	run_names.remove(run)
	subprocess.call('cp /home/ph290/data1/qump_out_python/annual_means/*'+run+'* /home/ph290/data1/qump_out_python/annual_means/use/', shell=True)


for run in run_names:
	subprocess.call('cp /home/ph290/data1/qump_out_python/annual_means/*'+run+'* /home/ph290/data1/qump_out_python/annual_means/dont_use/', shell=True)

'''
