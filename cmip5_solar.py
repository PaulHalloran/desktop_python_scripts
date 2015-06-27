import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# font = {'family' : 'normal',
# 'weight' : 'bold',
# 'size' : 12}

# matplotlib.rc('font', **font)
matplotlib.rcParams.update({'font.size': 15,'weight' : 'bold'})

file = '/home/ph290/data1/cmip5/forcing_data/solar_forcing_1610_2008.txt'

data = np.genfromtxt(file,skip_header = 3)

plt.plot(data[:,0],data[:,2])
plt.ylim([0,1400])
plt.xlim([1860,2000])
plt.xlabel('year')
plt.ylabel('Total Solar Irradiance, top of atm., (Wm$^{-2}$)')
plt.tight_layout()
plt.savefig('/home/ph290/Documents/teaching/masters/solar_1.png')

plt.plot(data[:,0],data[:,2])
plt.ylim([1365,1367])
plt.xlim([1860,2000])
plt.xlabel('year')
plt.ylabel('Total Solar Irradiance, top of atm., (Wm$^{-2}$)')
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
#plt.show()
plt.savefig('/home/ph290/Documents/teaching/masters/solar_2.png')
