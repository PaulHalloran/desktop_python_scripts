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

markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

styles = markers + [
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']


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
Read in subpolar gyre co2 flux data from QUMP
'''

run_names_order=str.split(line,' ')
run_names_order=run_names_order[0:-1]

input_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)

model_vars=['stash_30249']


dir_name2=['/home/ph290/data1/qump_out_python/annual_means/','/home/ph290/data1/qump_out_python/annual_means/low_pass_2/','/home/ph290/data1/qump_out_python/annual_means/band_pass_2/']
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
simulations with various smoothings...
'''

dir_name='/data/data1/ph290/boxmodel_testing_less_boxes_4_annual_matlab_low_pass/results/'
filenames=glob.glob(dir_name+'box_model*.csv')
input1=np.genfromtxt(filenames[0], delimiter=",")


filenamesc=glob.glob(dir_name+'box_model_qump_results_1_*.csv')
filenamesb=glob.glob(dir_name+'box_model_qump_results_?_1.csv')
file_order=np.empty(np.size(filenames))
file_order2=np.empty(np.size(filenames))

box_co2_flux=np.zeros((array_size,np.size(no_filenames),np.size(filenamesc),np.size(filenamesb)))
box_years=input1[:,0]

for i in range(np.size(filenames)):
    tmp=filenames[i].split('_')
    tmp2=tmp[12].split('.')
    file_order[i]=np.int(tmp2[0])
    tmp3=tmp[13].split('.')
    file_order2[i]=np.int(tmp3[0])

for i in np.arange(np.size(filenamesb)):
    for j in np.arange(np.size(filenamesc)):
        loc=np.where((file_order == i+1) & (file_order2 == j+1))
        #print 'reading box model file '+str(i)
        filenames[loc[0]]
        input1=np.genfromtxt(filenames[loc[0]], delimiter=",")
        box_co2_flux[:,:,j,i]=input1[:,1:np.size(no_filenames)+1]
    
'''
Now compare the two above sets of data to get a handle
on what drives the low and high frequency variability
'''

########################
# low pass #
########################

#overplot on each of the above the box model result with either low/band pass input variables.

plot_names=['1 no smoothing','all low-pass filtered','all low-pass filtered, salt constant','all low-pass filtered, temperature constant','all low-pass filtered, alkalinity constant','all low-pass filtered, atm. CO2 constant','all low-pass filtered, MOC constant']


for i in range(25):
	fig = plt.figure(figsize=(5*3.13,3*3.13))
	ax1 = fig.add_subplot(1,1,1)
	lns=plt.plot(qump_year,qump_data[1,:,i],c='b',linewidth=2,label='QUMP flux')
	for k in np.arange(np.size(plot_names)-1)+1:
		#ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k])
		lns += ln_tmp
		#this should plot out the variously filtered (at input to box model) box model results,
		#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux
	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs).draw_frame(False)
	plt.tight_layout()
	plt.show()

colours=['b','g','r','c','k']
sym_size_var=0.1
line_width_var=0.1












fig = plt.figure(figsize=(5*3.13,3*3.13))
for i,k in enumerate(np.append([0],np.arange(6,11))):
    ax1 = fig.add_subplot(2,4,i+1)
    for j in np.arange(np.size(filenamesb)):
        ax1.scatter(qump_data[0,:,:],(box_co2_flux[:,:,k,j]/1.12e13)*1.0e15/12.0, s=sym_size_var, facecolor='1.0',color=colours[j], lw = line_width_var)
        plt.plot([-10,10],[-10,10])
        plt.xlim(-5,10)
        plt.ylim(-5,10)
        plt.title(plot_names[k])
        plt.xlabel('ESM data')
        plt.ylabel('Box model data')

plt.tight_layout()
plt.show()
	
#details needed for band pass filtering data so that we are comparing just the variability rather than the longer then change
N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
low_cutoff=20.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
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

fig = plt.figure(figsize=(5*3.13,3*3.13))
for i,k in enumerate(np.arange(12,19)):  
    ax1 = fig.add_subplot(2,4,i+1)
    for j in np.arange(np.size(filenamesb)):
        tmp_shape=box_co2_flux[:,:,k,j].shape
        for l in range(tmp_shape[1]):
            dat=((box_co2_flux[:,l,k,j]/1.12e13)*1.0e15/12.0)
            dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
            dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
            #mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))

            dat2=qump_data[2,:,l]
            dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
            dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
            #mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))

            ax1.scatter(dat2_band_pass,dat_band_pass, s=sym_size_var, facecolor='1.0',color=colours[j], lw = line_width_var)
            plt.plot([-10,10],[-10,10])
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.title(plot_names[k])
            plt.xlabel('ESM data')
            plt.ylabel('Box model data')
plt.tight_layout()
plt.show()



lns=[]
ln_tmp=[]


fig = plt.figure(figsize=(30,20))

start=12
for i in (np.arange(6)+start):
	ax1 = fig.add_subplot(2,3,i-start)

	lns=plt.plot(qump_year,qump_data[0,:,i],c='b',linewidth=2,label='QUMP flux')
	#or k in np.array([0,7,8,9,10,19,20,21]):
	for k in np.array([0,19,20,21]):
		#ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k])
		lns += ln_tmp
		#this should plot out the variously filtered (at input to box model) box model results,
		#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

labs = [l.get_label() for l in lns]
plt.legend(lns, labs).draw_frame(False)


plt.tight_layout()
plt.show()
