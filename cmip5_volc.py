import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# font = {'family' : 'normal',
# 'weight' : 'bold',
# 'size' : 12}

# matplotlib.rc('font', **font)
matplotlib.rcParams.update({'font.size': 15,'weight' : 'bold'})

fig, ax1 = plt.subplots()
ax1.plot(data[:,0],data[:,1],'k')
plt.xlabel('year')
plt.ylabel('stratospheric aersol optical depth')
plt.ylim([0.0,0.6])

ax2 = ax1.twinx()
ax2.plot(data2[:,0],data2[:,1],'w')
# plt.ylim([1365,1367])
plt.xlim([1860,2000])
plt.ylim([-1.0,0.6])
plt.ylabel('global average surface temperature anomaly (K)')
ax2.yaxis.label.set_color('white')
ax2.tick_params(axis='y', colors='white')

ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
#plt.show()
plt.savefig('/home/ph290/Documents/teaching/masters/volc1.png')

'''
'''

fig, ax1 = plt.subplots()
ax1.plot(data[:,0],data[:,1],'k')
plt.xlabel('year')
plt.ylabel('stratospheric aersol optical depth')
plt.ylim([0.0,0.6])

ax2 = ax1.twinx()
ax2.plot(data2[:,0],data2[:,1],'r')
# plt.ylim([1365,1367])
plt.xlim([1860,2000])
plt.ylim([-1.0,0.6])
plt.ylabel('global average surface temperature anomaly (K)')
ax2.yaxis.label.set_color('red')

ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
#plt.show()
plt.savefig('/home/ph290/Documents/teaching/masters/volc2.png')
