
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
from scipy.stats import gaussian_kde

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

files = glob.glob('/home/ph290/data1/qump_out_python/annual_means/dont_use/*')
file_names = []
for file in files:
	file_names.append(file.split('_')[7])

file_names = np.unique(file_names)

run_names_order2 = []
run_names_order_index = []
for i,run in enumerate(run_names_order):
	if run in file_names:
		run_names_order2.append(run)
		run_names_order_index.append(run)

run_names_order = run_names_order2

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
	for i in range(np.size(run_names_order)):
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
#Read in data from box model:
'''

dir_name='/home/ph290/box_modelling/boxmodel_6_box_back_to_basics_tuning_hald_and_half/resultsb_predicted/'


parameter_sets = np.array([511+1,491+1,479+1])
np_parameter_sets = np.size(parameter_sets)

no_runs = 13



input1 = np.genfromtxt(dir_name+'spg_box_model_qump_results_1_'+str(parameter_sets[0])+'.csv', delimiter=",")
corcoeff_spg = np.zeros([np_parameter_sets,np.shape(input1)[1]])
corcoeff_spg[:] = np.NAN
rmse_spg = corcoeff_spg.copy()
corcoeff_stg = np.zeros([np_parameter_sets,np.shape(input1)[1]])
corcoeff_stg[:] = np.NAN
rmse_stg = corcoeff_stg.copy()
corcoeff_south = np.zeros([np_parameter_sets,np.shape(input1)[1]])
corcoeff_south[:] = np.NAN
rmse_south = corcoeff_south.copy()

from sklearn.metrics import mean_squared_error
from math import sqrt

'''

for i in range(np_parameter_sets):
	input1 = np.genfromtxt(dir_name+'spg_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
	input2 = np.genfromtxt(dir_name+'stg_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
	input3 = np.genfromtxt(dir_name+'south_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
	input1b = np.zeros([np.size(qump_year),no_runs])
	input1b[:] = np.NAN
	input2b = np.zeros([np.size(qump_year),no_runs])
	input2b[:] = np.NAN
	input3b = np.zeros([np.size(qump_year),no_runs])
	input3b[:] = np.NAN
	years = input1[:,0]
	for run in range(no_runs):
		for j,year in enumerate(qump_year):
			loc = np.where(years == year)
			input1b[j,run] = np.mean(input1[loc,run+1])
			input2b[j,run] = np.mean(input2[loc,run+1])
			input3b[j,run] = np.mean(input3[loc,run+1])
	for run in range(no_runs):
		corcoeff_spg[i,run] = np.corrcoef(qump_data[0,:,run]-scipy.stats.nanmean(qump_data[0,:,run]),input1b[:,run]-scipy.stats.nanmean((input1b[:,run]/1.12e13)*1.0e15/12.0))[0,1]
		corcoeff_stg[i,run] = np.corrcoef(qump_data_stg[0,:,run]-scipy.stats.nanmean(qump_data_stg[0,:,run]),input2b[:,run]-scipy.stats.nanmean((input2b[:,run]/1.12e13)*1.0e15/12.0))[0,1]
		corcoeff_south[i,run] = np.corrcoef(qump_data_south[0,:,run]-scipy.stats.nanmean(qump_data_south[0,:,run]),input3b[:,run]-scipy.stats.nanmean((input3b[:,run]/1.12e13)*1.0e15/12.0))[0,1]
                y_actual = qump_data[0,:,run]-scipy.stats.nanmean(qump_data[0,:,run])
                y_predicted = ((input1b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input1b[:,run]/1.12e13)*1.0e15/12.0)
                loc = np.where((np.isfinite(y_actual)) & (np.isfinite(y_predicted)))
                y_actual = y_actual[loc]-np.mean(y_actual[loc])
                y_predicted = y_predicted[loc]-np.mean(y_predicted[loc])
                rmse_spg[i,run] = sqrt(mean_squared_error(y_actual,y_predicted))
                y_actual = qump_data_stg[0,:,run]-scipy.stats.nanmean(qump_data_stg[0,:,run])
                y_predicted = ((input2b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input2b[:,run]/1.12e13)*1.0e15/12.0)
                loc = np.where((np.isfinite(y_actual)) & (np.isfinite(y_predicted)))
                y_actual = y_actual[loc]-np.mean(y_actual[loc])
                y_predicted = y_predicted[loc]-np.mean(y_predicted[loc])
                rmse_stg[i,run] = sqrt(mean_squared_error(y_actual,y_predicted))
                y_actual = qump_data_south[0,:,run]-scipy.stats.nanmean(qump_data_south[0,:,run])
                y_predicted = ((input3b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input3b[:,run]/1.12e13)*1.0e15/12.0)
                loc = np.where((np.isfinite(y_actual)) & (np.isfinite(y_predicted)))
                y_actual = y_actual[loc]-np.mean(y_actual[loc])
                y_predicted = y_predicted[loc]-np.mean(y_predicted[loc])
                rmse_south[i,run] = sqrt(mean_squared_error(y_actual,y_predicted))





overall_coreff = []
for i in range(np_parameter_sets):
	overall_coreff.append(np.mean([scipy.stats.nanmean(corcoeff_spg[i,:]),scipy.stats.nanmean(corcoeff_stg[i,:]),scipy.stats.nanmean(corcoeff_south[i,:])]))

'''

'''
i=0

for run in range(10,13):
	input1 = np.genfromtxt(dir_name+'spg_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
	input2 = np.genfromtxt(dir_name+'stg_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
	input3 = np.genfromtxt(dir_name+'south_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
	plt.plot(qump_year,qump_data[0,:,run]-scipy.stats.nanmean(qump_data[0,:,run]),label = 'QUMP')
	plt.plot(qump_year,((input1b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input1b[:,run]/1.12e13)*1.0e15/12.0),label = 'Box')
	plt.title('SPG')
	plt.legend()
	plt.show()


	plt.plot(qump_year,qump_data_stg[0,:,run]-scipy.stats.nanmean(qump_data_stg[0,:,run]),label = 'QUMP')
	plt.plot(qump_year,((input2b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input2b[:,run]/1.12e13)*1.0e15/12.0),label = 'Box')
	plt.title('Eq')
	plt.legend()
	plt.show()


	plt.plot(qump_year,qump_data_south[0,:,run]-scipy.stats.nanmean(qump_data_south[0,:,run]),label = 'QUMP')
	plt.plot(qump_year,((input3b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input3b[:,run]/1.12e13)*1.0e15/12.0),label = 'Box')
	plt.title('south')
	plt.legend()
	plt.show()
'''

'''
#produce scatter plot
'''
#f, axarr = plt.subplots(1, 1)
#f.set_size_inches(2,3)
plt.figure(num=None, figsize=(5, 5))
i=0

input1 = np.genfromtxt(dir_name+'spg_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
input2 = np.genfromtxt(dir_name+'stg_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
input3 = np.genfromtxt(dir_name+'south_box_model_qump_results_'+str(i+1)+'_'+str(parameter_sets[i])+'.csv', delimiter=",")
input1b = np.zeros([np.size(qump_year),no_runs])
input1b[:] = np.NAN
input2b = np.zeros([np.size(qump_year),no_runs])
input2b[:] = np.NAN
input3b = np.zeros([np.size(qump_year),no_runs])
input3b[:] = np.NAN
years = input1[:,0]
for run in range(no_runs):
        for j,year in enumerate(qump_year):
                loc = np.where(years == year)
                input1b[j,run] = np.mean(input1[loc,run+1])
                input2b[j,run] = np.mean(input2[loc,run+1])
                input3b[j,run] = np.mean(input3[loc,run+1])


x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []

for run in range(13):
        x1.extend(qump_data[0,:,run]-scipy.stats.nanmean(qump_data[0,:,run]))
        y1.extend(((input1b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input1b[:,run]/1.12e13)*1.0e15/12.0))

        # x2.extend(qump_data_stg[0,:,run]-scipy.stats.nanmean(qump_data_stg[0,:,run]))
        # y2.extend(((input2b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input2b[:,run]/1.12e13)*1.0e15/12.0))

        # x3.extend(qump_data_south[0,:,run]-scipy.stats.nanmean(qump_data_south[0,:,run]))
        # y3.extend(((input3b[:,run]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((input3b[:,run]/1.12e13)*1.0e15/12.0))


xb = np.array(x1)
xb = np.ma.masked_invalid(xb)
yb = np.array(y1)
yb = np.ma.masked_invalid(yb)
xb.mask = xb.mask | yb.mask
xb.mask = xb.mask | yb.mask
xb = xb[np.logical_not(xb.mask)]
yb = yb[np.logical_not(yb.mask)]

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xb, yb)
print r_value*r_value

xy = np.vstack([xb,yb]) 
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')
#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xb, yb)
#axarr[0,box].text(0.8,0.8,'R$^2$ = '+str(r_value**2))
plt.ylim(-2,3)
plt.xlim(-2,3)
plt.xlabel('Earth System Model CO$_2$ flux\n(mol-C m$^{-2}$ yr$^{-1}$)')
plt.ylabel('Box model northern box CO$_2$ flux\n(mol-C m$^{-2}$ yr$^{-1}$)')

# #bottom left
# axarr[2,0].set_ylabel('Southern box\nBox model CO$_2$ flux\n(mol-C m$^{-2}$ yr$^{-1}$)', fontsize=12)

# #left middle
# axarr[1,0].set_ylabel('Low latitude box\nBox model CO$_2$ flux\n(mol-C m$^{-2}$ yr$^{-1}$)', fontsize=12)

# #left top
# axarr[0,0].set_ylabel('Northern box\nBox model CO$_2$ flux\n(mol-C m$^{-2}$ yr$^{-1}$)', fontsize=12)

# #left bottom
# axarr[2,0].set_xlabel('Earth System Model CO$_2$ flux\n(mol-C m$^{-2}$ yr$^{-1}$)', fontsize=12)

# #bottom middle
# axarr[2,1].set_xlabel('Earth System Model CO$_2$ flux\n(mol-C m$^{-2}$ yr$^{-1}$)', fontsize=12)

# #bottom right
# axarr[2,2].set_xlabel('Earth System Model CO$_2$ flux\n(mol-C m$^{-2}$ yr$^{-1}$)', fontsize=12)

# #top left
# axarr[0,0].set_title('Tuned to northern box', fontsize=12)

# #top middle
# axarr[0,1].set_title('Tuned to low latitude box', fontsize=12)

# #top right
# axarr[0,2].set_title('Tuned to southern box', fontsize=12)

plt.tight_layout()
plt.savefig('/home/ph290/Documents/figures/half_and_half_prediction.png')
#plt.show()
