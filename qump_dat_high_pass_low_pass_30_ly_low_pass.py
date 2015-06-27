import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
import matplotlib.mlab as ml
from scipy.interpolate import griddata

def rmse(ts1,ts2):
    #ts1 = predicted value
    #ts2 = true value
    arraysize=np.size(ts1)
    diff_sq=np.square(ts1-ts2)
    mse=np.sum(diff_sq)*(1.0/arraysize)
    return np.sqrt(mse)

array_size=239

source="/home/ph290/data1/boxmodel_testing_less_boxes_4_annual_matlab_high_low_pass/results/order_runs_processed_in_box_model.txt"
inp = open(source,"r")
count=0
for line in inp:
    count +=1

'''
Read in QUMP data
'''

run_names_order=str.split(line,' ')
run_names_order=run_names_order[0:-1]

input_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)

model_vars=['stash_101','stash_102','stash_103','stash_104','stash_200','stash_30249','moc_stm_fun']


dir_name2='/home/ph290/data1/qump_out_python/annual_means/'
dir_name3='/home/ph290/data1/qump_out_python_30_yr/annual_means/'
no_filenames=glob.glob(dir_name2+'*30249.txt')
filenames2=glob.glob(dir_name2+'*'+run_names_order[0]+'*'+model_vars[0]+'.txt')

input2=np.genfromtxt(filenames2[0], delimiter=",")
no_time_series=input2[0,:].size-1

qump_year=np.zeros(array_size)
qump_data=np.zeros((np.size(model_vars),no_time_series,array_size,np.size(no_filenames)))

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

file_names=[]
for i in range(np.size(no_filenames)):
	filenames2=glob.glob(dir_name2+'*'+run_names_order[i]+'*'+model_vars[0]+'.txt')
	tmp=str(filenames2).split('_')
	file_names.append('_'.join(tmp[5:7]))
	
	
'''
Analysis and low/high-pass filtering
'''

N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
low_cutoff=30.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
high_cutoff=1.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
middle_cuttoff_low=1.0
middle_cuttoff_high=5.0

Wn_low=timestep_between_values/low_cutoff
Wn_high=timestep_between_values/high_cutoff
Wn_mid_low=timestep_between_values/middle_cuttoff_low
Wn_mid_high=timestep_between_values/middle_cuttoff_high


#design butterworth filters - or if want can replace butter with bessel
b, a = scipy.signal.butter(N, Wn_low, btype='low')
b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')

for k,model_var_cnt in enumerate(model_vars):
	for i in range(np.size(no_filenames)):
		qump_low_pass=np.zeros((np.size(qump_data[k,:,0,i]),np.size(qump_data[k,0,:,i])))
		qump_band_pass=np.zeros((np.size(qump_data[k,:,0,i]),np.size(qump_data[k,0,:,i])))
		qump_band_pass_around_zero=np.zeros((np.size(qump_data[k,:,0,i]),np.size(qump_data[k,0,:,i])))
		for j in range(np.size(qump_data[k,:,0,i])):
			qump_low_pass[j] = scipy.signal.filtfilt(b, a, qump_data[k,j,:,i])
			qump_middle_tmp=scipy.signal.filtfilt(b1, a1, qump_data[k,j,:,i])
			qump_band_pass[j]=scipy.signal.filtfilt(b2, a2, qump_middle_tmp)+np.mean(qump_data[k,j,0:19,i])
			qump_band_pass_around_zero[j]=scipy.signal.filtfilt(b2, a2, qump_middle_tmp)
		out=np.empty((qump_low_pass.shape[0]+1,qump_low_pass.shape[1]))
		out[0,:]=qump_year
		out[1:,:]=qump_low_pass
		np.savetxt(dir_name3+'low_pass_2/qump_data_'+file_names[i]+'_'+model_var_cnt+'.txt', np.transpose(out), delimiter=',')
		out[1:,:]=qump_band_pass
                #print np.mean(qump_band_pass)
		np.savetxt(dir_name3+'/band_pass_2/qump_data_'+file_names[i]+'_'+model_var_cnt+'.txt', np.transpose(out), delimiter=',')
		out[1:,:]=qump_band_pass_around_zero
		np.savetxt(dir_name3+'/band_pass_2_around_zero/qump_data_'+file_names[i]+'_'+model_var_cnt+'.txt', np.transpose(out), delimiter=',')

                if (model_var_cnt == model_vars[5]) & (i == 0):

                    plt.plot(qump_year,qump_data[k,0,:,i])
                    plt.plot(qump_year,qump_low_pass[0,:])

plt.title(' pass')
plt.show()

'''
And now for testing, I'm just smoothing the N. SPG box to see if the results look more like what we previously saw
'''


for k,model_var_cnt in enumerate(model_vars):
	for i in range(np.size(no_filenames)):
		qump_low_pass=np.zeros((np.size(qump_data[k,:,0,i]),np.size(qump_data[k,0,:,i])))
		qump_band_pass=np.zeros((np.size(qump_data[k,:,0,i]),np.size(qump_data[k,0,:,i])))
		qump_band_pass_around_zero=np.zeros((np.size(qump_data[k,:,0,i]),np.size(qump_data[k,0,:,i])))
		for j in range(np.size(qump_data[k,:,0,i])):
                    qump_low_pass[j] = qump_data[k,j,:,i]
                    qump_band_pass[j] = qump_data[k,j,:,i]
                    qump_band_pass_around_zero[j] = qump_data[k,j,:,i]

                    if j == 0:
                        qump_low_pass[j] = scipy.signal.filtfilt(b, a, qump_data[k,j,:,i])
                        qump_middle_tmp=scipy.signal.filtfilt(b1, a1, qump_data[k,j,:,i])
                        qump_band_pass[j]=scipy.signal.filtfilt(b2, a2, qump_middle_tmp)+np.mean(qump_data[k,j,0:19,i])
                        qump_band_pass_around_zero[j]=scipy.signal.filtfilt(b2, a2, qump_middle_tmp)

		out=np.empty((qump_low_pass.shape[0]+1,qump_low_pass.shape[1]))
		out[0,:]=qump_year
		out[1:,:]=qump_low_pass
		np.savetxt(dir_name3+'low_pass_3/qump_data_'+file_names[i]+'_'+model_var_cnt+'.txt', np.transpose(out), delimiter=',')
		out[1:,:]=qump_band_pass
                #print np.mean(qump_band_pass)
		np.savetxt(dir_name3+'band_pass_3/qump_data_'+file_names[i]+'_'+model_var_cnt+'.txt', np.transpose(out), delimiter=',')
		out[1:,:]=qump_band_pass_around_zero
		np.savetxt(dir_name3+'band_pass_3_around_zero/qump_data_'+file_names[i]+'_'+model_var_cnt+'.txt', np.transpose(out), delimiter=',')

                if (model_var_cnt == model_vars[5]) & (i == 0):

                    plt.plot(qump_year,qump_data[k,0,:,i])
                    plt.plot(qump_year,qump_low_pass[0,:])

plt.title(' pass')
plt.show()
