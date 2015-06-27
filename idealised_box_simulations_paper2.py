import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

results = np.genfromtxt('/home/ph290/box_modelling/box_model_obs_paper/results/spg_box_model_qump_results_1.csv',delimiter = ',')
results_stg = np.genfromtxt('/home/ph290/box_modelling/box_model_obs_paper/results/stg_box_model_qump_results_1.csv',delimiter = ',')

forcing_dir = '/home/ph290/box_modelling/box_model_obs_paper/forcing_data/co2/'

co2_tmp = np.genfromtxt(forcing_dir+'co2_a.txt',delimiter = ',')

co2 = np.zeros([co2_tmp.shape[0],4])

co2[:,0] = np.genfromtxt(forcing_dir+'co2_a.txt',delimiter = ',')[:,1]
co2[:,1] = np.genfromtxt(forcing_dir+'co2_b.txt',delimiter = ',')[:,1]
co2[:,2] = np.genfromtxt(forcing_dir+'co2_c.txt',delimiter = ',')[:,1]
co2[:,3] = np.genfromtxt(forcing_dir+'co2_d.txt',delimiter = ',')[:,1]

mpl.rcdefaults()

font = {'family' : 'monospace',
        'weight' : 'bold',
        'family' : 'serif',
        'size'   : 14}

mpl.rc('font', **font)

'''
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 4))

for i in range(4):
  ax1.plot(co2[:,i],linewidth = 6,alpha= 0.4)

ax1.set_ylabel('atm. CO$_2$ (ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)

for i in range(4):
  ax2.plot(results[:,i+1],linewidth = 6,alpha= 0.4)

ax2.set_ylim([40,180])
ax2.set_ylabel('atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax2.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)

#plt.arrow(0,0,0,1, shape='full', lw=3, length_includes_head=True, head_width=.01)


a1 = matplotlib.patches.Arrow(0.5-0.05,0.5,0.1,0.0, width=0.8,edgecolor='none',facecolor='gray',fill=True,transform=fig.transFigure, figure=fig,alpha=0.25)

fig.lines.extend([a1])
fig.canvas.draw()

plt.tight_layout()
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_1.png')
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_1.pdf')
plt.show(block = False)
plt.close('all')
'''

'''
spg-stg difference plots
'''


#for i in range(4):
#  ax1.plot(co2[:,i],linewidth = 6,alpha= 0.4)
#
#ax1.set_ylabel('atm. CO$_2$ (ppm)', multialignment='center',fontweight='bold',fontsize = 14)
#ax1.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)

plt.close('all')

colours = ['b','r']

i=2

fig, (ax1) = plt.subplots(1,1,figsize=(5, 4))
ax1.plot(results[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '-',color=colours[0])
ax1.plot(results_stg[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '--',color=colours[1])

ax1.set_xlim([0,100])
min1 = 40
max1 = 150
ax1.set_ylim([min1,max1])
ax1.set_ylabel('atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_ylabel('Subtropical atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)

plt.tight_layout()
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_2.png')
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_2.pdf')
plt.show(block = False)
#plt.close('all')

'''
2
'''

fig, (ax1) = plt.subplots(1,1,figsize=(5, 4))
ax1.plot(results[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '-',color=colours[0])
ax1.plot(results_stg[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '--',color=colours[1])

ax1.set_xlim([100,200])
min1 = 100
max1 = 160
ax1.set_ylim([min1,max1])
ax1.set_ylabel('atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_ylabel('Subtropical atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)

plt.tight_layout()
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_2.png')
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_2.pdf')
plt.show(block = False)
#plt.close('all')
