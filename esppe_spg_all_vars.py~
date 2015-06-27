import numpy as np
import glob
import matplotlib.pyplot as plt 

directory = '/home/ph290/data1/qump_out_python/annual_means/'

t_files = glob.glob(directory+'*101.txt')
alk_files = glob.glob(directory+'*104.txt')

t_yr = []
t_data = []
alk_yr = []
alk_data = []

for file in t_files:
    data = np.genfromtxt(file, delimiter=",")
    yr = data[:,0]
    spg = data[:,1]
    t_yr.append(yr)
    t_data.append(spg)

for file in alk_files:
    data = np.genfromtxt(file, delimiter=",")
    yr = data[:,0]
    spg = data[:,1]
    alk_yr.append(yr)
    alk_data.append(spg)

alpha_val = 0.4
lw = 3

plt.close('all')
fig = plt.figure(figsize=(10, 5))
fig.add_subplot(121)
for i,data in enumerate(t_data):
    plt.plot(t_yr[i],data - np.mean(data[0:20]),alpha = alpha_val,linewidth=lw)

plt.xlim(1860,2100)
plt.locator_params(axis = 'x', nbins = 4)
plt.locator_params(axis = 'y', nbins = 4)
plt.xlabel('year')
plt.ylabel('Temperature anomaly (K)')

fig.add_subplot(122)
for i,data in enumerate(alk_data):
    plt.plot(t_yr[i],data - np.mean(data[0:20]),alpha = alpha_val,linewidth=lw)

plt.xlim(1860,2100)
plt.locator_params(axis = 'x', nbins = 4)
plt.locator_params(axis = 'y', nbins = 4)
plt.xlabel('year')
plt.ylabel('Alkalinty anomaly ($\mu$mol kg$^{-1}$)')

plt.tight_layout()
# plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/n_atl/figure10_jul14.png')



