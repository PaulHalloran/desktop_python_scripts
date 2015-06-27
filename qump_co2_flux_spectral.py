# from numpy import sin, linspace, pi
# from pylab import plot, show, title, xlabel, ylabel, subplot
# from scipy import fft, arange
# import numpy as np
# import glob
# import matplotlib.pyplot as plt
# import matplotlib
# import scipy
# 
# '''
# Read in QUMP data
# '''
# 
# array_size=239
# 
# source="/home/ph290/data1/boxmodel_testing_less_boxes_4_annual_matlab_high_low_pass/results/order_runs_processed_in_box_model.txt"
# inp = open(source,"r")
# count=0
# for line in inp:
#     count +=1
# 
# 
# run_names_order=str.split(line,' ')
# run_names_order=run_names_order[0:-1]
# 
# input_year=np.zeros(array_size)
# qump_co2_flux=np.zeros(array_size)
# 
# model_vars=['stash_101','stash_102','stash_103','stash_104','stash_200','stash_30249','moc_stm_fun']
# 
# 
# dir_name2='/home/ph290/data1/qump_out_python/annual_means/'
# no_filenames=glob.glob(dir_name2+'*30249.txt')
# filenames2=glob.glob(dir_name2+'*'+run_names_order[0]+'*'+model_vars[0]+'.txt')
# 
# input2=np.genfromtxt(filenames2[0], delimiter=",")
# no_time_series=input2[0,:].size-1
# 
# qump_year=np.zeros(array_size)
# qump_data=np.zeros((np.size(model_vars),no_time_series,array_size,np.size(no_filenames)))
# #variable,box,years,ensemble no.
# 
# input2=np.genfromtxt(filenames2[0], delimiter=",")
# qump_year=input2[:,0]
# 
# for k,model_var_cnt in enumerate(model_vars):
# 	for i in range(np.size(no_filenames)):
# 		filenames2=glob.glob(dir_name2+'*'+run_names_order[i]+'*'+model_var_cnt+'.txt')
# 		input2=np.genfromtxt(filenames2[0], delimiter=",")
# 		no_time_series=input2[0,:].size-1
# 		if input2[:,1].size == array_size:
# 			for j in range(no_time_series):
# 				qump_data[k,j,:,i]=input2[:,j+1]
#                                 #variable,box no. (spg is 0),year,ens. member

y=[]
for i in np.arange(np.size(qump_data[5,0,0,:])):
    data=qump_data[5,0,:,i]
    test=np.isnan(data)
    test2=np.where(test == True)
    if np.size(test2) == 0:
    	y=np.append(y,data)

n=np.size(y)
Y = fft(y)/n
Y = Y[range(n/2)]

Fs = 1000
k = arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax1.plot(y)
ax2=fig.add_subplot(2,1,2)
ax2.plot(frq,abs(Y))
ax2.set_ylim(0,1)
ax2.set_yscale('log')
plt.show()

