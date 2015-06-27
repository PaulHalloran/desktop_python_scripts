

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
from matplotlib.colors import LinearSegmentedColormap

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
#Read in subpolar gyre co2 flux data from QUMP
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

#testing...
for k in np.arange(np.size(dir_name2)):
    for i in range(np.size(no_filenames)):
        filenames2=glob.glob(dir_name2[k]+'*'+run_names_order[i]+'*'+model_vars[0]+'.txt')
        input2=np.genfromtxt(filenames2[0], delimiter=",")
	if input2[:,1].size == array_size:
            qump_data[k,:,i]=qump_data[k,:,i]-scipy.stats.nanmean(qump_data[k,:,i])


    
		
'''
#Read in subpolar gyre co2 flux data from box model
#simulations with various smoothings...
'''

dir_name='/home/ph290/box_modelling/boxmodel_6_box_back_to_basics_tuning/resultsb_new/'
#/home/ph290/box_modelling/boxmodel_6_box_filtered_vars/results/'
filenames=glob.glob(dir_name+'spg_box_model*.csv')
input1=np.genfromtxt(filenames[0], delimiter=",")


filenamesc=glob.glob(dir_name+'spg_box_model_qump_results_1_*.csv')
filenamesb=glob.glob(dir_name+'spg_box_model_qump_results_?_1.csv')
file_order=np.empty(np.size(filenames))
file_order2=np.empty(np.size(filenames))

box_co2_flux=np.zeros((array_size,np.size(no_filenames),np.size(filenamesc),np.size(filenamesb)))
box_years=input1[:,0]

for i in range(np.size(filenames)):
    tmp=filenames[i].split('_')
    tmp2=tmp[-2]
    file_order[i]=np.int(tmp2)
    tmp3=tmp[-1].split('.')
    file_order2[i]=np.int(tmp3[0])

for i in np.arange(np.size(filenamesb)):
    for j in np.arange(np.size(filenamesc)):
        loc=np.where((file_order == i+1) & (file_order2 == j+1))
        #print 'reading box model file '+str(i)
        #filenames[loc[0]]
        #print filenames[loc[0]]
        input1=np.genfromtxt(filenames[loc[0]], delimiter=",")
        years = input1[:,0]
        years2 = np.unique(np.floor(years))
        tmp = input1[:,1:np.size(no_filenames)+1]
        tmp2 = np.empty([239, 26])
        for k,year in enumerate(range(1860,2099)):
            loc = np.where(np.floor(years) == year)
            tmp2[k] = scipy.stats.nanmean(tmp[loc,:],axis = 1)
        box_co2_flux[:,:,j,i]=tmp2


#tetsing....
shape_tmp=box_co2_flux.shape
for i in np.arange(shape_tmp[1]):
    for j in np.arange(shape_tmp[2]):
        for k in np.arange(shape_tmp[3]):
            box_co2_flux[:,i,j,k]=box_co2_flux[:,i,j,k]-scipy.stats.nanmean(box_co2_flux[:,i,j,k])

'''
#Now compare the two above sets of data to get a handle
#on what drives the low and high frequency variability
'''

########################
# low pass # band pass #
########################

#overplot on each of the above the box model result with either low/band pass input variables.

plot_names=['1 no smoothing','2 salt smoothed (all boxes - yes to start with) (band pass)','3 temp smoothed (band pass)','4 Alk smoothed (band)','5 moc smoothed (band pass)','6 atm co2 smoothed (band pass)','7 alk amd co2 (band pass)','8 all band passed','9 all band passed, salt const','10 all band passed, temp const','11 all band passed, alk const','12 all band passed, moc const','13 all band passed, temp and alk const','14 alk, co2 and T (band pass)','15 All band passed','16 no smoothing, salt const','17 no smoothing, temp const','18 no smoothing, alk const','19 no smoothing, moc const','20 no smoothing, atm co2 const','21 no smoothing temp and alk const']

for i in range(1):
	fig = plt.figure(figsize=(5*3.13,3*3.13))
	ax1 = fig.add_subplot(3,1,1)
	lns=plt.plot(qump_year,qump_data[0,:,i],c='b',linewidth=2,label='QUMP flux')
	for k in np.append(np.arange(0,6),[11]):
		#ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,label=plot_names[k])
		lns += ln_tmp
		#this should plot out the variously filtered (at input to box model) box model results,
		#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs).draw_frame(False)

	ax2 = fig.add_subplot(3,1,2)
	lns=plt.plot(qump_year,qump_data[0,:,i],c='b',linewidth=2,label='QUMP flux')
	for k in np.arange(7):
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
		lns += ln_tmp
		#this should plot out the variously filtered (at input to box model) box model results,
		#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux
	
	plt.plot(qump_year,qump_data[0,:,i])

	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs).draw_frame(False)

	ax3 = fig.add_subplot(3,1,3)
	lns=plt.plot(qump_year,qump_data[2,:,i]-scipy.stats.nanmean(qump_data[2,:,i]),c='b',linewidth=2,label='QUMP flux')
	for k in np.arange(7,13):
		ln_tmp=plt.plot(qump_year,((box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0)-scipy.stats.nanmean((box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0),marker=styles[k],markersize=4,label=plot_names[k])
		lns += ln_tmp
		#this should plot out the variously filtered (at input to box model) box model results,
		#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux
	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs).draw_frame(False)

	plt.tight_layout()
	plt.show()
	#plt.savefig('detete.png')

colours=['b','g','r','c','k']
sym_size_var=0.1
line_width_var=0.1

fig = plt.figure(figsize=(5*3.13,3*3.13))
for i,k in enumerate(np.append(np.arange(7),[13])):
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
#plt.savefig('detete.png')

'''
#Figure 9
'''

plot_names=['1 no smoothing','salinity','temperature','alkalinity','MOC','atm. CO$_2$','alk amd co2','8 all band passed','9 all band passed, salt const','10 all band passed, temp const','11 all band passed, alk const','12 all band passed, moc const','13 all band passed, ','atm. CO$_2$, alk. and temp.','15 All band passed','16 no smoothing, salt const','17 no smoothing, temp const','18 no smoothing, alk const','19 no smoothing, moc const','20 no smoothing, atm co2 const','21 no smoothing temp and alk const']


plt.close('all')
colours=['k','b','g','r','c']
sym_size_var=5.0
line_width_var=5
alpha_val = 1.0


marker_sty = np.array(['|', 'x', '_', '+','1', 'p', '3', '2', '4', 'H', 'v', '8', '<', '>'])
lns=[]
ln_tmp=[]



cdict1 = {'red':  ((0.0, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
        }

cdict2 = {'red':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),
        }
        
cdict3 = {'red':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
        }

                   
cdict4 = cdict1.copy()
cdict4['alpha'] = ((0.0, 0.0, 0.0),
                   (1.0, 0.3, 0.3))
                   
RedAlpha = LinearSegmentedColormap('RedAlpha', cdict4)
plt.register_cmap(cmap=RedAlpha)

cdict5 = cdict2.copy()
cdict5['alpha'] = ((0.0, 0.0, 0.0),
                   (1.0, 0.3, 0.3))
                   
BlueAlpha = LinearSegmentedColormap('BlueAlpha', cdict5)
plt.register_cmap(cmap=BlueAlpha)

cdict6 = cdict3.copy()
cdict6['alpha'] = ((0.0, 0.0, 0.0),
                   (1.0, 0.3, 0.3))
                   
GreenAlpha = LinearSegmentedColormap('GreenAlpha', cdict6)
plt.register_cmap(cmap=GreenAlpha)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
plt.plot([-10,10],[-10,10],'k')
k = np.append(np.arange(7),[13])
i=0
maps = ['RedAlpha','GreenAlpha','BlueAlpha']

# for j in np.arange(np.size(filenamesb)):
for j in np.arange(3):
	cmap=plt.get_cmap(maps[j])
	x = qump_data[0,:,:]
	y = (box_co2_flux[:,:,k[0],j]/1.12e13)*1.0e15/12.0
	loc = np.where(np.logical_not(np.isnan(x)) & np.logical_not(np.isnan(y)))
	x = x[loc]
	y = y[loc]
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	ln_tmp = ax1.scatter(x, y, c=z,cmap = cmap, s=10, edgecolor='',label=plot_names[k[0]])
	#ln_tmp = ax1.plot(qump_data[0,:,:],(box_co2_flux[:,:,k[0],j]/1.12e13)*1.0e15/12.0,color=colours[j],alpha = alpha_val,marker=marker_sty[j], linestyle='None',label=plot_names[k[0]])
	plt.xlim(-2,2)
	plt.ylim(-2,2)
	plt.xlabel('Earth System Model\nCO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
	plt.ylabel('Box model (no variables smoothed)\nCO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
	lns.append(ln_tmp)
	plt.locator_params(axis = 'x', nbins = 4)
	plt.locator_params(axis = 'y', nbins = 4)

#labs = [l.get_label() for l in lns]
#plt.legend(lns, labs,loc=2,prop={'size':9.6}).draw_frame(False)


alpha_val = 1.0

colours2=['#000000','#ff0000','#00ff00','#0000ff','#ff00ff','#ffff00','#0ffff0']
lns=[]
ln_tmp=[]

order=[3,5,2,1,4,13]

ax1 = fig.add_subplot(1,2,2)
plt.plot([-10,10],[-10,10],'k')
for i,k in enumerate(order):
	ln_tmp=[]
	#for j in np.arange(np.size(filenamesb)):
	j=0
	#ax1.scatter(((box_co2_flux[:,:,k,j]/1.12e13)*1.0e15/12.0),((box_co2_flux[:,:,0,j]/1.12e13)*1.0e15/12.0), s=sym_size_var,color=colours2[i], lw = line_width_var,alpha = alpha_val, facecolors='none')
	ln_tmp = ax1.plot(((box_co2_flux[:,:,0,j]/1.12e13)*1.0e15/12.0),((box_co2_flux[:,:,k,j]/1.12e13)*1.0e15/12.0),colours2[i],alpha = alpha_val,marker='o',markersize=5, linestyle='None',label=plot_names[k])
	ln_tmp = ln_tmp[0]
	plt.xlim(-6,6)
	plt.ylim(-6,6)
	plt.xlabel('Box model (no variables smoothed)\nCO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
	plt.ylabel('Box model (variables smoothed)\nCO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
	lns.append(ln_tmp)
	plt.locator_params(axis = 'x', nbins = 4)
	plt.locator_params(axis = 'y', nbins = 4)
	
labs = [l.get_label() for l in lns]
legend = plt.legend(lns, labs,loc=3,prop={'size':9.6},title='Low-pass filtered variable:',ncol=2)
plt.setp(legend.get_title(),fontsize='small')
legend.get_frame().set_alpha(0.75)

fig.text(0.1, 0.9, 'a', ha='center', va='center',size = 'x-large')
fig.text(0.58, 0.9, 'b', ha='center', va='center',size = 'x-large')

	
plt.tight_layout()
plt.show(block = True)
#plt.savefig('/home/ph290/Documents/figures/n_atl/figure9_may15.png')

	
'''
#high-pass filtered:
'''	
	
	
# #details needed for band pass filtering data so that we are comparing just the variability rather than the longer then change
# N=5.0
# #I think N is the order of the filter - i.e. quadratic
# timestep_between_values=1.0 #years valkue should be '1.0/12.0'
# low_cutoff=20.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
# high_cutoff=1.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
# middle_cuttoff_low=1.0
# middle_cuttoff_high=5.0
# 
# Wn_low=timestep_between_values/low_cutoff
# Wn_high=timestep_between_values/high_cutoff
# Wn_mid_low=timestep_between_values/middle_cuttoff_low
# Wn_mid_high=timestep_between_values/middle_cuttoff_high
# 
# #design butterworth filters - or if want can replace butter with bessel
# b, a = scipy.signal.butter(N, Wn_low, btype='low')
# b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
# b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')
# 
# #plot_names=['1 no smoothing','2 salt smoothed (all boxes - yes to start with) (band pass)','3 temp smoothed (band pass)','4 Alk smoothed (band)','5 moc smoothed (band pass)','6 atm co2 smoothed (band pass)','7 alk amd co2 (band pass)','8 all band passed','9 all band passed, salt const','10 all band passed, temp const','11 all band passed, alk const','12 all band passed, moc const','13 all band passed, temp and alk const','14 alk, co2 and T (band pass)','15 All band passed','16 no smoothing, salt const','17 no smoothing, temp const','18 no smoothing, alk const','19 no smoothing, moc const','20 no smoothing, atm co2 const','21 no smoothing temp and alk const']
# 
# fig = plt.figure(figsize=(5*3.13,3*3.13))
# for i,k in enumerate(np.arange(14,21)):  
#     ax1 = fig.add_subplot(2,4,i+1)
#     for j in np.arange(np.size(filenamesb)):
#         tmp_shape=box_co2_flux[:,:,k,j].shape
#         for l in range(tmp_shape[1]):
#             dat=((box_co2_flux[:,l,k,j]/1.12e13)*1.0e15/12.0)
#             dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
#             dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
#             #mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))
#             dat2=qump_data[2,:,l]
#             dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
#             dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
#             #mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))
#             ax1.scatter(dat2_band_pass,dat_band_pass, s=sym_size_var, facecolor='1.0',color=colours[j], lw = line_width_var)
#             plt.plot([-10,10],[-10,10])
#             plt.xlim(-2,2)
#             plt.ylim(-2,2)
#             plt.title(plot_names[k])
#             plt.xlabel('ESM data')
#             plt.ylabel('Box model data')
# 
# plt.tight_layout()
# plt.show()
# 
# ----------------
# 
# #details needed for band pass filtering data so that we are comparing just the variability rather than the longer then change
# N=5.0
# #I think N is the order of the filter - i.e. quadratic
# timestep_between_values=1.0 #years valkue should be '1.0/12.0'
# low_cutoff=20.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
# high_cutoff=1.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
# middle_cuttoff_low=1.0
# middle_cuttoff_high=5.0
# 
# Wn_low=timestep_between_values/low_cutoff
# Wn_high=timestep_between_values/high_cutoff
# Wn_mid_low=timestep_between_values/middle_cuttoff_low
# Wn_mid_high=timestep_between_values/middle_cuttoff_high
# 
# #design butterworth filters - or if want can replace butter with bessel
# b, a = scipy.signal.butter(N, Wn_low, btype='low')
# b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
# b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')
# 
# #plot_names=['1 no smoothing','2 salt smoothed (all boxes - yes to start with) (band pass)','3 temp smoothed (band pass)','4 Alk smoothed (band)','5 moc smoothed (band pass)','6 atm co2 smoothed (band pass)','7 alk amd co2 (band pass)','8 all band passed','9 all band passed, salt const','10 all band passed, temp const','11 all band passed, alk const','12 all band passed, moc const','13 all band passed, temp and alk const','14 alk, co2 and T (band pass)','15 All band passed','16 no smoothing, salt const','17 no smoothing, temp const','18 no smoothing, alk const','19 no smoothing, moc const','20 no smoothing, atm co2 const','21 no smoothing temp and alk const']
# 
# fig = plt.figure(figsize=(5*3.13,3*3.13))
# for i,k in enumerate(np.arange(14,21)):  
#     ax1 = fig.add_subplot(2,4,i+1)
#     for j in np.arange(np.size(filenamesb)):
#         tmp_shape=box_co2_flux[:,:,k,j].shape
#         for l in range(tmp_shape[1]):
#             dat=((box_co2_flux[:,l,k,j]/1.12e13)*1.0e15/12.0)
#             dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
#             dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
#             #mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))
#             dat2=qump_data[2,:,l]
#             dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
#             dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
#             #mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))
#             ax1.scatter(dat2_band_pass,dat_band_pass, s=sym_size_var, facecolor='1.0',color=colours[j], lw = line_width_var)
#             plt.plot([-10,10],[-10,10])
#             plt.xlim(-2,2)
#             plt.ylim(-2,2)
#             plt.title(plot_names[k])
#             plt.xlabel('ESM data')
#             plt.ylabel('Box model data')
# 
# plt.tight_layout()
# plt.show()





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

plt.close('all')
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
plt.plot([-10,10],[-10,10],'k')
k = np.append(np.arange(7),[13])
i=0
maps = ['RedAlpha','GreenAlpha','BlueAlpha']

# for j in np.arange(np.size(filenamesb)):
for j in np.arange(3):
	tmp_shape=box_co2_flux[:,:,k,j].shape
	for l in np.arange(tmp_shape[1]): 
		if not(l == 11):
			dat=((box_co2_flux[:,l,14,j]/1.12e13)*1.0e15/12.0)
			dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
			dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
			#mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))
			dat2=qump_data[2,:,l]
			dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
			dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
			#mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))
			cmap=plt.get_cmap(maps[j])
			x = dat2_band_pass
			y = dat_band_pass
			loc = np.where(np.logical_not(np.isnan(x)) & np.logical_not(np.isnan(y)))
			x = x[loc]
			y = y[loc]
			xy = np.vstack([x,y])
			z = gaussian_kde(xy)(xy)
			ln_tmp = ax1.scatter(x, y, c=z,cmap = cmap, s=10, edgecolor='',label=plot_names[k[0]])
			#ln_tmp = ax1.plot(qump_data[0,:,:],(box_co2_flux[:,:,k[0],j]/1.12e13)*1.0e15/12.0,color=colours[j],alpha = alpha_val,marker=marker_sty[j], linestyle='None',label=plot_names[k[0]])
			plt.xlim(-1,1)
			plt.ylim(-1,1)
			plt.xlabel('High-pass filtered ESM\nCO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
			plt.ylabel('Box model (all high-pass filtered)\nCO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
			lns.append(ln_tmp)
			plt.locator_params(axis = 'x', nbins = 4)
			plt.locator_params(axis = 'y', nbins = 4)

#labs = [l.get_label() for l in lns]
#plt.legend(lns, labs,loc=2,prop={'size':9.6}).draw_frame(False)



alpha_val = 1.0


black= '#000000'
red= '#ff0000'
green= '#00ff00'
blue= '#0000ff'
mouve= '#ff00ff'
yellow= '#ffff00'

colours2=[green,blue,mouve,black,red,yellow]

lns=[]
ln_tmp=[]

order=[16,15,18,17,19,20]

plot_names[20] = 'temp. + alk.'
plot_names = ['1 no smoothing', 'salinity', 'temperature', 'alkalinity', 'MOC', 'atm. CO$_2$', 'alk amd co2', '8 all band passed', '9 all band passed, salt const', '10 all band passed, temp const', '11 all band passed, alk const', '12 all band passed, moc const', '13 all band passed, ', 'atm. CO$_2$, alk. and temp.', '15 All band passed', 'salinity', 'temperature', 'alkalinity', 'MOC', 'atm. CO$_2$', 'temp. + alk.']

ax1 = fig.add_subplot(1,2,2)
plt.plot([-10,10],[-10,10],'k')
for i,k in enumerate(order):
	ln_tmp=[]
	#for j in np.arange(np.size(filenamesb)):
	j=0
	tmp_shape=box_co2_flux[:,:,k,j].shape
	for l in np.arange(tmp_shape[1]): 
		if not(l == 11):
			dat=((box_co2_flux[:,l,k,j]/1.12e13)*1.0e15/12.0)
			dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
			dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
			#mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))
			dat2=qump_data[2,:,l]
			dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
			dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
			#mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))
			cmap=plt.get_cmap(maps[j])
			x = dat2_band_pass
			y = dat_band_pass
			ln_tmp = ax1.plot(x,y,colours2[i],alpha = alpha_val,marker='o',markersize=5, linestyle='None',label=plot_names[k])
			plt.xlim(-2,2)
			plt.ylim(-4,4)
			plt.xlabel('Box model (all high-pass filtered)\nCO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
			plt.ylabel('Box model (constant variable)\nCO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
			plt.locator_params(axis = 'x', nbins = 4)
			plt.locator_params(axis = 'y', nbins = 4)
	ln_tmp = ln_tmp[0]
	lns.append(ln_tmp)	
	
	
labs = [l.get_label() for l in lns]
legend = plt.legend(lns, labs,loc=3,prop={'size':9.6},title='Constant variable:',ncol=2)
plt.setp(legend.get_title(),fontsize='x-small')
legend.get_frame().set_alpha(0.75)

fig.text(0.12, 0.9, 'a', ha='center', va='center',size = 'x-large')
fig.text(0.6, 0.9, 'b', ha='center', va='center',size = 'x-large')

	
plt.tight_layout()
plt.show(block = False)
#plt.savefig('/home/ph290/Documents/figures/n_atl/figure11_may15.png')

'''

'''
#calculating ensemble member RMSEs or should I do R2s?
'''



r2=np.zeros(tmp_shape[1])
r2.fill(np.nan)

for i in np.arange(tmp_shape[1]):
    qump=qump_data[0,:,i]
    box=(box_co2_flux[:,i,0,0]/1.12e13)*1.0e15/12.0
    tmp=np.where(np.logical_not(np.isnan(box)))
    qump=qump[tmp]
    box=box[tmp]
    if qump.size <> 0:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(qump,box)
        r2[i]=r_value**2
  
r2_sorted=np.sort(r2)
r2_sorted=r2_sorted[np.logical_not(np.isnan(r2_sorted))]

no=3
worst=np.zeros(no)
best=np.zeros(no)

for i in np.arange(np.size(best)):
    worst[i]=np.where(r2 == r2_sorted[i])[0]
    best[i]=np.where(r2 == r2_sorted[-1*(i+1)])[0]





lns=[]
ln_tmp=[]
alpha_val = 0.4
lw = 2

titlesa = ['highest R$^2$','2$^{nd}$ highest R$^2$','3$^{rd}$ highest R$^2$',]
titlesb = ['lowest R$^2$','2$^{nd}$ lowest R$^2$','3$^{rd}$ lowest R$^2$',]

plot_names=['none', 'salinity', 'temperature', 'alkalinity', 'MOC', 'atm. CO$_2$', 'alk amd co2', 'no variables constant', '9 all band passed, salt const', 'temp. constant', 'alk. constant', 'MOC constant', 'tmp. and alk. constant', 'atm. CO$_2$, alk. and temp.', '15 All band passed', 'salinity', 'temperature', 'alkalinity', 'MOC', 'atm. CO$_2$', 'temp. + alk.']

plt.close('all')
fig = plt.figure(figsize=(10, 8))

for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,no,i+1)
	lns=plt.plot(qump_year,qump_data[0,:,best[i]],c='b',linewidth=2,label='ESM')	
	for k in np.array([0,2,3,5]):
		plt.plot(qump_year,(box_co2_flux[:,best[i],k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k],alpha = alpha_val,linewidth=lw)
		plt.title(titlesa[i])
        plt.xlim(1860,2100)
        plt.ylim(-2,4)
        plt.locator_params(axis = 'x', nbins = 4)
        plt.locator_params(axis = 'y', nbins = 4)

labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=2,bbox_to_anchor=(-2.4,1.0),prop={'size':12}).draw_frame(False)

lns=[]
ln_tmp=[]

for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,no,i+1+np.size(best))
	plt.plot(qump_year,qump_data[0,:,worst[i]],c='b',linewidth=2,label='ESM')
	for k in np.array([0,2,3,5]):
		ln_tmp=plt.plot(qump_year,(box_co2_flux[:,worst[i],k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k],alpha = alpha_val,linewidth=lw)
		plt.title(titlesb[i])
		if i == 0:
			lns += ln_tmp
        plt.xlim(1860,2100)
        plt.locator_params(axis = 'x', nbins = 4)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.ylim(-2,4)

labs = [l.get_label() for l in lns]
legend1 = plt.legend(lns, labs, loc=2,bbox_to_anchor=(-2.4,0.9),prop={'size':12}).draw_frame(False)
fig.text(0.24, 0.43, 'Box model filtered input:', ha='center', va='center',size = 'small')



fig.text(0.07, 0.5, 'CO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)', ha='center', va='center', rotation='vertical',size = 'large')
fig.text(0.5, 0.05, 'Year', ha='center', va='center',size = 'large')



#plt.tight_layout()
plt.show(block = False)
#plt.savefig('/home/ph290/Documents/figures/n_atl/figure6_jul14.png')



'''
#multmodel mean
'''
tmp_shape=box_co2_flux.shape
box_co2_flux_mean=np.zeros([tmp_shape[0],tmp_shape[2],tmp_shape[3]])

for i1 in np.arange(tmp_shape[3]):
    for i2 in np.arange(tmp_shape[2]):
        for i3 in np.arange(tmp_shape[0]):
            tmp=box_co2_flux[i3,:,i2,i1]
	    tmp=tmp[np.logical_not(np.isnan(tmp))]
            box_co2_flux_mean[i3,i2,i1]=scipy.stats.nanmean(tmp)


qump_co2_flux_mean=np.zeros(tmp_shape[0])

for i in np.arange(tmp_shape[0]):
    tmp=qump_data[0,i,:]
    tmp=tmp[np.logical_not(np.isnan(tmp))]
    qump_co2_flux_mean[i]=scipy.stats.nanmean(tmp)

lns=[]
ln_tmp=[]


fig = plt.figure(figsize=(30,20))

start=5

lns=plt.plot(qump_year,qump_co2_flux_mean,c='b',linewidth=2,label='QUMP flux')
#or k in np.array([0,7,8,9,10,19,20,21]):
for k in np.array([0,3,5]):
	ln_tmp=plt.plot(qump_year,(box_co2_flux_mean[:,k,0]/1.12e13)*1.0e15/12.0,label=plot_names[k])
	lns += ln_tmp
	#this should plot out the variously filtered (at input to box model) box model results,
	#for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0).draw_frame(False)


plt.tight_layout()
#plt.show()
plt.savefig('detete.png')


print 'this is using the simulation with everything varying as normal, but one variabile held constant at its mean value from the 1st 20 years'

lns=[]
ln_tmp=[]

tmp_shape=box_co2_flux.shape
l=0

fig = plt.figure(figsize=(30,20))

start=5
for i in (np.arange(6)+start):

        dat2=qump_data[0,:,i]
        dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
        dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
        mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))

	ax1 = fig.add_subplot(2,3,i-start)

	lns=plt.plot(qump_year,mdat2,c='b',linewidth=2,label='QUMP flux')
	for k in [0]:
#np.append([0],np.arange(15,21)):

            

            dat=((box_co2_flux[:,i,k,l]/1.12e13)*1.0e15/12.0)
            dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
            dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
            mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))



            #ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
            ln_tmp=plt.plot(qump_year,mdat,label=plot_names[k])
            lns += ln_tmp
            #this should plot out the variously filtered (at input to box model) box model results,
            #for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0).draw_frame(False)


plt.tight_layout()
#plt.show()
plt.savefig('detete.png')


print 'this version of the plot uses teh analsysis done with proper band-pass filtered data all the way...'

lns=[]
ln_tmp=[]

tmp_shape=box_co2_flux.shape
l=0

fig = plt.figure(figsize=(30,20))

start=0
for i in (np.arange(6)+start):

        dat2=qump_data[2,:,i]
        #dat2_tmp=scipy.signal.filtfilt(b1, a1, dat2)
        #dat2_band_pass=scipy.signal.filtfilt(b2, a2, dat2_tmp)
        #mdat2=np.ma.masked_array(dat2_band_pass,np.isnan(dat2_band_pass))

	ax1 = fig.add_subplot(2,3,i-start)

	lns=plt.plot(qump_year,dat2,c='b',linewidth=2,label='QUMP flux')
	for k in np.append(np.arange(7,13),[14]):

            

            dat=((box_co2_flux[:,i,k,l]/1.12e13)*1.0e15/12.0)
            #dat_tmp=scipy.signal.filtfilt(b1, a1, dat)
            #dat_band_pass=scipy.signal.filtfilt(b2, a2, dat_tmp)
            #mdat=np.ma.masked_array(dat_band_pass,np.isnan(dat_band_pass))



            #ln_tmp=plt.plot(qump_year,(box_co2_flux[:,i,k]/1.12e13)*1.0e15/12.0,marker=styles[k],markersize=4,label=plot_names[k])
            ln_tmp=plt.plot(qump_year,dat,label=plot_names[k])
            lns += ln_tmp
            #this should plot out the variously filtered (at input to box model) box model results,
            #for the QUMP ensemble member specified by i. - should be able to spot which fail to match QUMP flux

labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0).draw_frame(False)
#plt.show()
plt.savefig('detete.png')

'''
#best and worst
'''



r2=np.zeros(tmp_shape[1])
r2.fill(np.nan)

for i in np.arange(tmp_shape[1]):
    qump=qump_data[2,:,i]
    box=(box_co2_flux[:,i,7,0]/1.12e13)*1.0e15/12.0
    tmp=np.where(np.logical_not(np.isnan(box)))
    qump=qump[tmp]
    box=box[tmp]
    if qump.size <> 0:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(qump,box)
        r2[i]=r_value**2
  
r2_sorted=np.sort(r2)
r2_sorted=r2_sorted[np.logical_not(np.isnan(r2_sorted))]

alpha_val = 0.4
lw = 3

no=1
worst=np.zeros(no)
best=np.zeros(no)

for i in np.arange(np.size(best)):
    worst[i]=np.where(r2 == r2_sorted[i])[0]
    best[i]=np.where(r2 == r2_sorted[-1*(i+1)])[0]

lns=[]
ln_tmp=[]


plot_names = ['1 no smoothing', 'salinity', 'temperature', 'alkalinity', 'MOC', 'atm. CO$_2$', 'alk amd co2', 'no variables constant', '9 all band passed, salt const', 'temp. constant', 'alk. constant', 'MOC constant', 'tmp. and alk. constant', 'atm. CO$_2$, alk. and temp.', '15 All band passed', 'salinity', 'temperature', 'alkalinity', 'MOC', 'atm. CO$_2$', 'temp. + alk.']

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)








plt.close('all')
fig = plt.figure(figsize=(16, 8))

for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,2,i+1)
	lns=plt.plot(qump_year,(qump_data[2,:,best[i]]),c='k',label='ESM',alpha = alpha_val,linewidth=lw)
        labs = [l.get_label() for l in lns]
        legend1 = plt.legend(lns, labs, loc=2, ncol = 3,prop={'size':14})
        legend1.get_frame().set_alpha(0.75)
	polygon = matplotlib.patches.Rectangle((2050,-1.0), 50, 2, color='k',alpha = 0.2)
	ax1.add_patch(polygon)

	for k in [7,9,10,12]:
		ln_tmp=plt.plot(qump_year,((box_co2_flux[:,best[i],k,0]/1.12e13)*1.0e15/12.0),label=plot_names[k],alpha = alpha_val,linewidth=lw)
		plt.title('Highest R$^2$')
		lns += ln_tmp
        plt.xlim(1860,2100)
        plt.ylim(-1,1)
        plt.locator_params(axis = 'x', nbins = 4)


for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,2,i+1+np.size(best))
	lns=plt.plot(qump_year,(qump_data[2,:,worst[i]]),c='k',label='ESM',alpha = alpha_val,linewidth=lw)
        lns=[]
        ln_tmp=[]
	polygon = matplotlib.patches.Rectangle((2050,-1.0), 50, 2, color='k',alpha = 0.2)
	ax1.add_patch(polygon)

	for k in [7,9,10,12]:
		ln_tmp=plt.plot(qump_year,((box_co2_flux[:,worst[i],k,0]/1.12e13)*1.0e15/12.0),label=plot_names[k],alpha = alpha_val,linewidth=lw)
		plt.title('Lowest R$^2$')
		lns += ln_tmp
        plt.xlim(1860,2100)
        plt.ylim(-1,1)
        plt.locator_params(axis = 'x', nbins = 4)



labs = [l.get_label() for l in lns]
legend2 = plt.legend(lns, labs, loc=3,ncol = 2,prop={'size':12},title='Box model, all input variables high-pass filtered:')
legend2.get_frame().set_alpha(0.75)

fig.text(0.035, 0.5, 'CO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)', ha='center', va='center', rotation='vertical',size = 'medium')

#######################

for i in np.arange(np.size(best)):
	ax1 = fig.add_subplot(2,2,i+1+2)
	lns=plt.plot(qump_year,(qump_data[2,:,best[i]]),c='k',label='ESM',alpha = alpha_val,linewidth=lw)
        labs = [l.get_label() for l in lns]
        #legend1 = plt.legend(lns, labs, loc=2, ncol = 3,prop={'size':14})
        #legend1.get_frame().set_alpha(0.75)

	for k in [7,9,10,12]:
		ln_tmp=ax1.plot(qump_year,((box_co2_flux[:,best[i],k,0]/1.12e13)*1.0e15/12.0),label=plot_names[k],alpha = alpha_val,linewidth=lw)
		ax1.set_title('Highest R$^2$')
		lns += ln_tmp
        ax1.set_xlim(2050,2100)
        ax1.set_ylim(-1,1)
        plt.locator_params(axis = 'x', nbins = 4)
	polygon = matplotlib.patches.Rectangle((2050,-1.0), 50, 2, color='k',alpha = 0.2)
	ax1.add_patch(polygon)


for i in np.arange(np.size(best)):
	ax2 = fig.add_subplot(2,2,i+1+np.size(best)+2)
	lns=ax2.plot(qump_year,(qump_data[2,:,worst[i]]),c='k',label='ESM',alpha = alpha_val,linewidth=lw)
        lns=[]
        ln_tmp=[]
	for k in [7,9,10,12]:
		ln_tmp=plt.plot(qump_year,((box_co2_flux[:,worst[i],k,0]/1.12e13)*1.0e15/12.0),label=plot_names[k],alpha = alpha_val,linewidth=lw)
		ax2.set_title('Lowest R$^2$')
		lns += ln_tmp
        ax2.set_xlim(2050,2100)
        ax2.set_ylim(-1,1)
        plt.locator_params(axis = 'x', nbins = 4)
	polygon = matplotlib.patches.Rectangle((2050,-1.0), 50, 2, color='k',alpha = 0.2)
	ax2.add_patch(polygon)

fig.text(0.5,0.035, 'Year', ha='center', va='center', rotation='horizontal',size = 'medium')

line_y1 = 0.465
line_y2 = 0.535

ax3 = plt.axes([0,0,1,1], axisbg=(1,1,1,0))
x,y = np.array([[0.13,0.40], [line_y1,line_y2]])
line1 = matplotlib.lines.Line2D(x, y, lw=5., color='k', alpha=0.4)
x,y = np.array([[0.475,0.475], [line_y1,line_y2]])
line2 = matplotlib.lines.Line2D(x, y, lw=5., color='k', alpha=0.4)

shift = 0.4225
x,y = np.array([[0.13+shift,0.40+shift], [line_y1,line_y2]])
line3 = matplotlib.lines.Line2D(x, y, lw=5., color='k', alpha=0.4)
x,y = np.array([[0.475+shift,0.475+shift], [line_y1,line_y2]])
line4 = matplotlib.lines.Line2D(x, y, lw=5., color='k', alpha=0.4)

ax3.add_line(line1)
ax3.add_line(line2)
ax3.add_line(line3)
ax3.add_line(line4)

plt.show(block = False)
#plt.savefig('/home/ph290/Documents/figures/n_atl/figure5_aug14.png')

'''
