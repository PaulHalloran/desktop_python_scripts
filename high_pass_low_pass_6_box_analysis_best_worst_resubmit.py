
'''
#6-box model high/low/band pass analysis
'''

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
import matplotlib.mlab as ml
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
import scipy.stats

'''
#Read in subpolar gyre co2 flux data from QUMP
'''
array_size=239

source="/home/ph290/data1/boxmodel_testing_less_boxes_4_annual_matlab_high_low_pass/results/order_runs_processed_in_box_model.txt"
inp = open(source,"r")
count=0
for line in inp:
    count +=1

run_names_order=str.split(line,' ')
run_names_order=run_names_order[0:-1]

input_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)

model_vars=['stash_30249']


dir_name2=['/home/ph290/data1/qump_out_python/annual_means/']
no_filenames=glob.glob(dir_name2[0]+'*30249.txt')
filenames2=glob.glob(dir_name2[0]+'*'+run_names_order[0]+'*'+model_vars[0]+'.txt')

input2=np.genfromtxt(filenames2[0], delimiter=",")

qump_year=np.zeros(array_size)
qump_data=np.zeros((3,array_size,np.size(no_filenames)))
qump_data_stg=np.zeros((3,array_size,np.size(no_filenames)))
qump_data_south=np.zeros((3,array_size,np.size(no_filenames)))
#first 3 referes to normal, low-pass, band pass

input2=np.genfromtxt(filenames2[0], delimiter=",")
qump_year=input2[:,0]

for k in np.arange(np.size(dir_name2)):
	for i in range(np.size(no_filenames)):
		filenames2=glob.glob(dir_name2[k]+'*'+run_names_order[i]+'*'+model_vars[0]+'.txt')
		input2=np.genfromtxt(filenames2[0], delimiter=",")
		if input2[:,1].size == array_size:
			qump_data[k,:,i]=input2[:,1]
                        qump_data_stg[k,:,i]=input2[:,4]
                        qump_data_south[k,:,i]=input2[:,3]
			#should this be column 1???? Is this box 1?

#testing...
#for k in np.arange(np.size(dir_name2)):
#    for i in range(np.size(no_filenames)):
#        filenames2=glob.glob(dir_name2[k]+'*'+run_names_order[i]+'*'+model_vars[0]+'.txt')
#        input2=np.genfromtxt(filenames2[0], delimiter=",")
#	if input2[:,1].size == array_size:
#            qump_data[k,:,i]=qump_data[k,:,i]-np.mean(qump_data[k,:,i])
#            qump_data_stg[k,:,i]=qump_data_stg[k,:,i]-np.mean(qump_data_stg[k,:,i])
#            qump_data_south[k,:,i]=qump_data_south[k,:,i]-np.mean(qump_data_south[k,:,i])
    
'''
Read in data from box model:
'''

dir_name='/home/ph290/box_modelling/boxmodel_6_box_back_to_basics/results7b/'
input1 = np.genfromtxt(dir_name+'spg_box_model_qump_results_1.csv', delimiter=",")
input2 = np.genfromtxt(dir_name+'stg_box_model_qump_results_1.csv', delimiter=",")
input3 = np.genfromtxt(dir_name+'south_box_model_qump_results_1.csv', delimiter=",")

plt.close('all')

fig, ax = plt.subplots(3, sharex=True,sharey=True)

for i in range(27):
    ax[0].plot(qump_year,qump_data[0,:,i],c='b',linewidth=0.2,label='QUMP flux')
    y1 = (input1[:,i+1]/1.12e13)*1.0e15/12.0
    #ax[0].plot(input2[:,0],y1-scipy.stats.nanmean(y1),'r',linewidth=0.2)
    ax[0].plot(input2[:,0],y1,'r',linewidth=0.2)

for i in range(27):
    ax[1].plot(qump_year,qump_data_stg[0,:,i],c='b',linewidth=0.2,label='QUMP flux')
    y2 = (input2[:,i+1]/1.12e13)*1.0e15/12.0
    #ax[1].plot(input2[:,0],y2-scipy.stats.nanmean(y2),'r',linewidth=0.2)
    ax[1].plot(input2[:,0],y2,'r',linewidth=0.2)

for i in range(27):
    ax[2].plot(qump_year,qump_data_south[0,:,i],c='b',linewidth=0.2,label='QUMP flux')
    y2 = (input3[:,i+1]/1.12e13)*1.0e15/12.0
    #ax[2].plot(input3[:,0],y2-scipy.stats.nanmean(y2),'r',linewidth=0.2)
    ax[2].plot(input3[:,0],y2,'r',linewidth=0.2)

plt.show(block = False)


for i in range(3):
    plt.plot(qump_year,qump_data[0,:,i],c='b',linewidth=2,label='QUMP flux')
    y2 = (input1[:,i+1]/1.12e13)*1.0e15/12.0
    #t convert from co2
    plt.plot(input1[:,0],y2-scipy.stats.nanmean(y2),'r')
    plt.show()

