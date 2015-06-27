'''
Figure just to show that the box model does a reasonable job
'''

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal as signal

def rmse(ts1,ts2):
    #ts1 = predicted value
    #ts2 = true value
    arraysize=np.size(ts1)
    diff_sq=np.square(ts1-ts2)
    mse=np.sum(diff_sq)*(1.0/arraysize)
    return np.sqrt(mse)
    


'''
Read in QUMP data
'''

array_size=239

# source="/home/ph290/data1/boxmodel_testing_less_boxes_4_annual_matlab_high_low_pass/results/order_runs_processed_in_box_model.txt"
source="/Users/ph290/Public/order_runs_processed_in_box_model.txt"

inp = open(source,"r")
count=0
for line in inp:
    count +=1

run_names_order=str.split(line,' ')
run_names_order=run_names_order[0:-1]

input_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)

model_vars=['stash_30249']

dir_name2=['/Users/ph290/Public/mo_data/qump_out_python/annual_means/']
no_filenames=glob.glob(dir_name2[0]+'*30249.txt')
filenames2=glob.glob(dir_name2[0]+'*'+run_names_order[0]+'*'+model_vars[0]+'.txt')

input2=np.genfromtxt(filenames2[0], delimiter=",")

qump_year=np.zeros(array_size)
qump_data=np.zeros((3,array_size,np.size(no_filenames)))
#first 3 referes to normal, low-pass, band pass

input2=np.genfromtxt(filenames2[0], delimiter=",")
qump_year=input2[:,0]

for k in np.arange(np.size(dir_name2)):
	for i in range(np.size(no_filenames)):
		filenames2=glob.glob(dir_name2[k]+'*'+run_names_order[i]+'*'+model_vars[0]+'.txt')
		input2=np.genfromtxt(filenames2[0], delimiter=",")
		if input2[:,1].size == array_size:
			qump_data[k,:,i]=input2[:,1]
			#should this be column 1???? Is this box 1?
		

'''
Read in subpolar gyre co2 flux data from box model
'''

#dir_name='/home/ph290//data1/boxmodel_testing_less_boxes_4_annual_matlab_high_low_pass/results/'
dir_name='/Users/ph290/Public/tmp_box_model_data/results/'
filenames=glob.glob(dir_name+'box_model*.csv')
input1=np.genfromtxt(filenames[0], delimiter=",")


filenamesc=glob.glob(dir_name+'box_model_qump_results_1_*.csv')
filenamesb=glob.glob(dir_name+'box_model_qump_results_?_2.csv')
file_order=np.empty(np.size(filenames))
file_order2=np.empty(np.size(filenames))

box_co2_flux=np.zeros((array_size,np.size(no_filenames),np.size(filenamesc),np.size(filenamesb)))
box_years=input1[:,0]

for i in range(np.size(filenames)):
    tmp=filenames[i].split('_')
    tmp2=tmp[7].split('.')
    file_order[i]=np.int(tmp2[0])
    tmp3=tmp[8].split('.')
    file_order2[i]=np.int(tmp3[0])

for i in np.arange(np.size(filenamesb)):
    for j in np.arange(np.size(filenamesc)):
        loc=np.where((file_order == i+1) & (file_order2 == j+1))
        #print 'reading box model file '+str(i)
        filenames[loc[0]]
        input1=np.genfromtxt(filenames[loc[0]], delimiter=",")
        box_co2_flux[:,:,j,i]=input1[:,1:np.size(no_filenames)+1]

size_tmp=np.size(qump_data[k,0,:])
rmse_results=np.zeros(size_tmp)

for i in np.arange(size_tmp):
    rmse_results[i]=rmse(qump_data[k,:,i],(box_co2_flux[:,i,0,0]/1.12e13)*1.0e15/12.0)
    test=box_co2_flux[:,i,0,0][np.logical_not(np.isnan(box_co2_flux[:,i,0,0]))]
    if np.size(test) < 238:
    	rmse_results[i]=np.nan
    	

sorted_rmse=np.sort(rmse_results)
sorted_rmse=sorted_rmse[np.logical_not(np.isnan(sorted_rmse))]

no=3
top_fits=np.zeros(no)
bottom_fits=np.zeros(no)
for i in np.arange(no):
	top_fits[i]=np.reshape(np.where(rmse_results == sorted_rmse[i]),1)
	bottom_fits[i]=np.reshape(np.where(rmse_results == sorted_rmse[-1*(i+1)]),1)

#top_fits=[0,1,2,3,4]
#bottom_fits=[5,6,7,8,9]

ln_0=1
ln_1=1
ln_2=0.25

fig=plt.figure(facecolor='white')
for i in np.arange(no):
	ax1=plt.subplot2grid((2,no),(0,i))
	ax1.set_title('best fit: '+np.str(i+1))
	ax1.set_ylabel('CO$_2$ flux (umol m$^{-1}$ yr$^{-1}$)')
	ax1.set_xlabel('year')
	for j in np.arange(np.size(box_co2_flux[0,0,0,:])):
		ln1=ax1.plot(qump_year,qump_data[k,:,top_fits[i]],'k',linewidth=ln_0,label='ESM results')
		if j > 0:
			ln3=ax1.plot(qump_year,(box_co2_flux[:,top_fits[i],0,j]/1.12e13)*1.0e15/12.0,'grey',linewidth=ln_2,label='Box model other parameter sets')
		if j == 0:
			ln2=ax1.plot(qump_year,(box_co2_flux[:,top_fits[i],0,j]/1.12e13)*1.0e15/12.0,'r',linewidth=ln_1,label='Box model best fit parameter set')
		ax1.set_ylim(0,8)
		
lns=ln1+ln2+ln3
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,fontsize='xx-small').draw_frame(False)
	
for i in np.arange(no):
	ax2=plt.subplot2grid((2,no),(1,i))
	ax2.set_title('worst fit: '+np.str(i+1))
	ax2.set_ylabel('CO$_2$ flux (umol m$^{-1}$ yr$^{-1}$)')
	ax2.set_xlabel('year')
	for j in np.arange(np.size(box_co2_flux[0,0,0,:])):
		ax2.plot(qump_year,qump_data[k,:,bottom_fits[i-no]],'k',linewidth=ln_0)
		if j > 0:
			ax2.plot(qump_year,(box_co2_flux[:,bottom_fits[i-no],0,j]/1.12e13)*1.0e15/12.0,'grey',linewidth=ln_2)
		if j == 0:
			ax2.plot(qump_year,(box_co2_flux[:,bottom_fits[i-no],0,j]/1.12e13)*1.0e15/12.0,'r',linewidth=ln_1)
		ax2.set_ylim(0,8)

plt.tight_layout()
plt.show()


# i=0
# plt.plot(qump_year,qump_data[k,:,i],)
# plt.plot(qump_year,(box_co2_flux[:,i,0,0]/1.12e13)*1.0e15/12.0)
# plt.show()


