import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal as signal
from matplotlib.ticker import MaxNLocator


'''
Read in QUMP data
'''

array_size=239



input_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)

model_vars=['stash_101','stash_102','stash_103','stash_104','stash_200','stash_30249','moc_stm_fun']


dir_name2='/Users/ph290/Public/mo_data/qump_out_python/annual_means/'
### CHANGE THIS WHEN ON DESKTOP $###
filenames=glob.glob(dir_name2+'*30249.txt')
filenames2=glob.glob(dir_name2+'*txt')

input2=np.genfromtxt(filenames[0], delimiter=",")
no_time_series=input2[0,:].size-1

qump_year=np.zeros(array_size)
qump_data=np.zeros((np.size(model_vars),no_time_series,array_size,np.size(filenames)))
#variable,box,years,ensemble no.

input2=np.genfromtxt(filenames2[0], delimiter=",")
qump_year=input2[:,0]

for k,model_var_cnt in enumerate(model_vars):
	filenames_tmp=glob.glob(dir_name2+'*'+model_vars[k]+'*')
	for i in range(np.size(filenames)):
		input2=np.genfromtxt(filenames_tmp[i], delimiter=",")
		no_time_series=input2[0,:].size-1
		if input2[:,1].size == array_size:
			for j in range(no_time_series):
				qump_data[k,j,:,i]=input2[:,j+1]
                                #variable,box no. (spg is 0),year,ens. member

spg_as_flux=qump_data[5,0,:,:]
shape_tmp=np.shape(spg_as_flux)
spg_as_flux_high=np.copy(spg_as_flux)
spg_as_flux_low=np.copy(spg_as_flux)

N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
low_cutoff=30.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
high_cutoff=5.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years

Wn_low=timestep_between_values/low_cutoff
Wn_high=timestep_between_values/high_cutoff

#design butterworth filters - or if want can replace butter with bessel
b, a = scipy.signal.butter(N, Wn_low, btype='low')
b1, a1 = scipy.signal.butter(N, Wn_high, btype='high')

high_pass_all=[]

for i in range(shape_tmp[1]):
    spg_as_flux_low[:,i]=scipy.signal.filtfilt(b, a, spg_as_flux[:,i])
    spg_as_flux_high[:,i]=scipy.signal.filtfilt(b1, a1, spg_as_flux[:,i])
    tmp=spg_as_flux_high[:,i]
    test=np.isnan(tmp)
    test2=np.where(test == True)
    if np.size(test2) == 0:
        high_pass_all=np.append(high_pass_all,spg_as_flux_high[:,i])
        #y=spg_as_flux_high[:,i]

sym_size_var=1.0
line_width_var=0.5

names=['eg 1','eg 2','eg 3','eg 4']

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

fig = plt.figure(facecolor='white',figsize=(8, 6), dpi=120)
ax=fig.add_subplot(2,1,1)
#ax.set_xlabel('year')
ax.set_ylabel('CO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
no=4
for i in range(no):
    ax1=fig.add_subplot(no,2,i+1)
    ax1.set_title(names[i])
    ln1=ax1.plot(qump_year,spg_as_flux[:,i],'k',label='raw flux')
    ln2=ax1.plot(qump_year,spg_as_flux_low[:,i],'b',label='low pass >30 yrs')
    ln4=ax1.plot(qump_year,spg_as_flux_high[:,i]+spg_as_flux_low[:,i],'r',label='low pass + high pass', lw = line_width_var)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4,prune='lower'))
    ax2 = ax1.twinx()
    #ax2.set_ylabel('band pass flux')
    ax2.tick_params(axis='y', colors='g')
    ln3=ax2.plot(qump_year,spg_as_flux_high[:,i],'g',label='high pass < 5 yrs', lw = line_width_var)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

lns=ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,bbox_to_anchor=(1.1,-0.5), ncol=np.size(lns),prop={'size':10}).draw_frame(False)

ax3=fig.add_subplot(2,2,3)
ax3.set_xlabel('year')
ax3.set_ylabel('CO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
ax3.set_title('low-pass filtered (> 30 yrs)')
for i in range(shape_tmp[1]):
    ax3.plot(qump_year,spg_as_flux_low[:,i],'b')
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6,prune='lower'))        

ax4=fig.add_subplot(2,2,4)
ax4.set_xlabel('year')
#ax4.set_ylabel('CO$_2$ flux (umol m$^{-2}$ yr$^{-1}$)')
ax4.set_title('high-pass filtered (< 5 yrs)')
for i in range(shape_tmp[1]):
    ax4.scatter(qump_year,spg_as_flux_high[:,i], s=sym_size_var, facecolor='1.0', lw = line_width_var,color='g')
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6,prune='lower'))
    
ax4.set_xlim(1850,2100)
ax4.set_ylim(-1.5,1.5)

plt.tight_layout()
plt.show()



