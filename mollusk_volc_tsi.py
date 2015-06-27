'''
mlr

'''

import numpy as np
import glob
import subprocess
import os
import matplotlib.pyplot as plt
import numpy.ma as ma
import running_mean
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d


#volcanic_data

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file1)
data3 = np.genfromtxt(file1)
data4 = np.genfromtxt(file1)
data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]
# data_tmp[:,2] = data3[:,1]
# data_tmp[:,3] = data4[:,1]
data = np.mean(data_tmp,axis = 1)
data_final = data1.copy()
volcanic_data = data_final



smoothing_years = 20.0

volcanic_data[:,1] = running_mean.running_mean(data_final[:,1],36.0*smoothing_years)
volcanic_data[:,1] = np.log(volcanic_data[:,1])
volcanic_data[:,1][np.where(np.isnan(volcanic_data[:,1]))] = scipy.stats.nanmean(volcanic_data[:,1])
volcanic_data[:,1][np.where(volcanic_data[:,0] > 1950)] = scipy.stats.nanmean(volcanic_data[:,1])

plt.close('all')
# fig0 = plt.figure(figsize=(6, 4), dpi=80)
# plt.plot(volcanic_data[:,0],volcanic_data[:,1],'r')
# plt.show(block=False)



#solar_data

file1 = '/home/ph290/data0/misc_data/last_millenium_solar/tsi_SBF_11yr.txt'
data1 = np.genfromtxt(file1,skip_header = 4)
solar_year = data1[:,0]
solar_data = data1[:,1]

# fig1 = plt.figure(figsize=(6, 4), dpi=80)
# plt.plot(solar_year,solar_data,'g')
# plt.show(block=False)

#mollusk data

reynolds_file = '/home/ph290/data0/reynolds/ultra_data.csv'
reynolds_data_tmp = np.genfromtxt(reynolds_file,skip_header = 1,delimiter = ',')
reynolds_data_tmp = np.flipud(reynolds_data_tmp)

# fig2 = plt.figure(figsize=(6, 4), dpi=80)
# plt.plot(reynolds_data_tmp[:,0],reynolds_data_tmp[:,1],'b')
# plt.show(block=False)

loc = np.where((reynolds_data_tmp[:,0] <= 1150) & (reynolds_data_tmp[:,0] >= 800))[0]
reynolds_data = reynolds_data_tmp[loc,:]


volc = np.interp(reynolds_data[:,0], volcanic_data[:,0], volcanic_data[:,1])
tsi = np.interp(reynolds_data[:,0], solar_year, solar_data)

volc[np.where(np.isnan(volc))] = scipy.stats.nanmean(volc)
tsi[np.where(np.isnan(tsi))] = scipy.stats.nanmean(tsi)

reynolds_data[:,1] = signal.detrend(reynolds_data[:,1])
volc = signal.detrend(volc)
tsi = signal.detrend(tsi)

#reynolds_data[:,1] = running_mean.running_mean(reynolds_data[:,1],2)
#reynolds_data[np.where(np.isnan(reynolds_data[:,1])),1] = np.mean(reynolds_data[np.where(np.logical_not(np.isnan(reynolds_data[:,1]))),1])

#tsi = running_mean.running_mean(tsi,smoothing_years)
#tsi[np.where(np.isnan(tsi))] = np.mean(tsi[np.where(np.logical_not(np.isnan(tsi)))])

x = np.array([volc,tsi])

y = reynolds_data[:,1]

n = np.max(x.shape)   

X = np.vstack([np.ones(n), x]).T

c,m1,m2 = np.linalg.lstsq(X, y)[0]


fig3 = plt.figure(figsize=(6, 5), dpi=80)
ax1 = fig3.add_subplot(111)
ax1.plot(reynolds_data[:,0],reynolds_data[:,1],'k',linewidth = 2,alpha = 0.5)
# ax1.plot(reynolds_data[:,0],m1*volc+m2*tsi+c,'r',linewidth = 5,alpha = 0.5)
ax1.set_ylabel('iceland d180')
ax2 = ax1.twinx()
ax2.plot(reynolds_data[:,0],volc,'b',linewidth = 5,alpha = 0.5)
ax2.set_ylim([-3,3])
ax2.set_ylabel('20 yr smoothed log volcanic AOD')

plt.savefig('/home/ph290/Documents/figures/dave_1.png')
# plt.show(block=False)

#volcanic_data

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file1)
data3 = np.genfromtxt(file1)
data4 = np.genfromtxt(file1)
data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]
# data_tmp[:,2] = data3[:,1]
# data_tmp[:,3] = data4[:,1]
data = np.mean(data_tmp,axis = 1)
data_final = data1.copy()
volcanic_data = data_final

smoothing_years = 20.0

volcanic_data[:,1] = running_mean.running_mean(data_final[:,1],36.0*smoothing_years)
volcanic_data[:,1] = np.log(volcanic_data[:,1])

#solar_data

file1 = '/home/ph290/data0/misc_data/last_millenium_solar/tsi_SBF_11yr.txt'
data1 = np.genfromtxt(file1,skip_header = 4)
solar_year = data1[:,0]
solar_data = data1[:,1]


#mollusk data

reynolds_file = '/home/ph290/data0/reynolds/ultra_data.csv'
reynolds_data_tmp = np.genfromtxt(reynolds_file,skip_header = 1,delimiter = ',')
reynolds_data_tmp = np.flipud(reynolds_data_tmp)
loc = np.where((reynolds_data_tmp[:,0] <= 1850) & (reynolds_data_tmp[:,0] >= 1300))[0]
reynolds_data = reynolds_data_tmp[loc,:]

volc = np.interp(reynolds_data[:,0], volcanic_data[:,0], volcanic_data[:,1])
tsi = np.interp(reynolds_data[:,0], solar_year, solar_data)

volc[np.where(np.isnan(volc))] = scipy.stats.nanmean(volc)
tsi[np.where(np.isnan(tsi))] = scipy.stats.nanmean(tsi)

reynolds_data[:,1] = signal.detrend(reynolds_data[:,1])
volc = signal.detrend(volc)
tsi = signal.detrend(tsi)

# reynolds_data[:,1] = running_mean.running_mean(reynolds_data[:,1],10)
# reynolds_data[np.where(np.isnan(reynolds_data[:,1])),1] = np.mean(reynolds_data[np.where(np.logical_not(np.isnan(reynolds_data[:,1]))),1])

#tsi = running_mean.running_mean(tsi,smoothing_years)
#tsi[np.where(np.isnan(tsi))] = np.mean(tsi[np.where(np.logical_not(np.isnan(tsi)))])

x = np.array([volc,tsi])

y = reynolds_data[:,1]

n = np.max(x.shape)   

X = np.vstack([np.ones(n), x]).T

c,m1,m2 = np.linalg.lstsq(X, y)[0]

fig4 = plt.figure(figsize=(11, 5), dpi=80)
ax1 = fig4.add_subplot(111)
ax1.plot(reynolds_data[:,0],reynolds_data[:,1],'k',linewidth = 2,alpha = 0.5)
# ax1.plot(reynolds_data[:,0],m1*volc+m2*tsi+c,'r',linewidth = 5,alpha = 0.5)
ax1.set_ylabel('iceland d180')
ax2 = ax1.twinx()
ax2.plot(reynolds_data[:,0],volc*-1,'b',linewidth = 5,alpha = 0.5)
ax2.set_ylim([-3,3])
ax2.set_ylabel('20 yr smoothed log volcanic AOD *-1')

plt.savefig('/home/ph290/Documents/figures/dave_2.png')
#plt.show(block = False)

print np.mean(reynolds_data[:,1])
print np.mean(m1*volc+m2*tsi+c)

