import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy

'''
Read in QUMP data
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

model_vars=['stash_101','stash_102','stash_103','stash_104','stash_200','stash_30249','moc_stm_fun']


dir_name2='/home/ph290/data1/qump_out_python/annual_means/'
no_filenames=glob.glob(dir_name2+'*30249.txt')
filenames2=glob.glob(dir_name2+'*'+run_names_order[0]+'*'+model_vars[0]+'.txt')

input2=np.genfromtxt(filenames2[0], delimiter=",")
no_time_series=input2[0,:].size-1

qump_year=np.zeros(array_size)
qump_data=np.zeros((np.size(model_vars),no_time_series,array_size,np.size(no_filenames)))
#variable,box,years,ensemble no.

input2=np.genfromtxt(filenames2[0], delimiter=",")
qump_year=input2[:,0]

for k,model_var_cnt in enumerate(model_vars):
	for i in range(np.size(no_filenames)):
		filenames2=glob.glob(dir_name2+'*'+run_names_order[i]+'*'+model_var_cnt+'.txt')
		input2=np.genfromtxt(filenames2[0], delimiter=",")
		no_time_series=input2[0,:].size-1
		if input2[:,1].size == array_size:
			for j in range(no_time_series):
				qump_data[k,j,:,i]=input2[:,j+1]
                                #variable,box no. (spg is 0),year,ens. member

fig=plt.figure()
ax1=fig.add_subplot(211)
var=3
for i in range(qump_data[3,0,0,:].size):
    ax1.plot(qump_year,qump_data[var,0,:,i]-np.mean(qump_data[var,0,0:19,i]))
ax2=fig.add_subplot(212)
var=1
for i in range(qump_data[3,0,0,:].size):
    ax2.plot(qump_year,qump_data[var,0,:,i]-np.mean(qump_data[var,0,3:19,i]))

plt.show()

var=1
var2=3
for i in range(qump_data[3,0,0,:].size):
    plt.plot(qump_data[var,0,:,i]-np.mean(qump_data[var,0,0:19,i]),qump_data[var2,0,:,i]-np.mean(qump_data[6,0,0:19,i]))
    plt.ylabel('AMOC anomaly (Sv)')
    plt.xlabel('SPG ALK anomaly (mmol/m3)')
    plt.show()

print 'I suspect that the non-linearity might well be because the alkalinity is not fully spun up. Of clourse this is not causal - does it come back to a raster MOC entraining alkalinity from the subtropical gyre? Now of course how important is the alkalinity in the turnover?'
