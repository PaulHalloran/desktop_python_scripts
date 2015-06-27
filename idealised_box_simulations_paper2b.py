import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

results = np.genfromtxt('/home/ph290/box_modelling/boxmodel_6_box_back_to_basics/results/spg_box_model_qump_results_3.csv',delimiter = ',')
results_stg = np.genfromtxt('/home/ph290/box_modelling/boxmodel_6_box_back_to_basics/results/stg_box_model_qump_results_3.csv',delimiter = ',')

forcing_dir = '/home/ph290/box_modelling/boxmodel_6_box_back_to_basics/forcing_data/co2/'

co2_tmp = np.genfromtxt(forcing_dir+'rcp85_1.txt',delimiter = ',')

co2 = np.zeros([co2_tmp.shape[0],4])

co2[:,0] = np.genfromtxt(forcing_dir+'rcp85_1.txt',delimiter = ',')[:,1]
co2[:,1] = np.genfromtxt(forcing_dir+'rcp85_2.txt',delimiter = ',')[:,1]
co2[:,2] = np.genfromtxt(forcing_dir+'rcp85_3.txt',delimiter = ',')[:,1]

rcp85_yr = np.genfromtxt(forcing_dir+'historical_and_rcp85_atm_co2.txt',delimiter = ',')[:,0]
rcp85 = np.genfromtxt(forcing_dir+'historical_and_rcp85_atm_co2.txt',delimiter = ',')[:,1]


mpl.rcdefaults()

font = {'family' : 'monospace',
        'weight' : 'bold',
        'family' : 'serif',
        'size'   : 14}

mpl.rc('font', **font)


plt.close('all')

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 4))

leg_lab = ['y = 1.0285**x +c','y = 1.0265**x +c','y = 1.0305**x +c']

for i in range(3):
  ax1.plot(co2[:,i],linewidth = 6,alpha= 0.4,label = leg_lab[i])

ax1.legend(loc = 2,prop={'size':10, 'family' : 'normal','weight' : 'bold'},ncol = 1).draw_frame(False)


#ax1.plot(rcp85_yr-1860,rcp85,'k',linewidth = 6,alpha= 0.4)

ax1.set_xlim([0,240])
ax1.set_ylim([200,1800])

ax1.set_ylabel('atm. CO$_2$ (ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)

for i in range(3):
  ax2.plot(results[:,0]-results[0,0],results[:,i+1],linewidth = 6,alpha= 0.4)

ax2b = ax2.twinx()
for i in range(3):
  ax2b.plot(results[:,0]-results[0,0],results_stg[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '--')

leg_lab2 = ['Subpolar N. Atlantic (left axis)','Subtropical/equatorial (right axis)']
tmp = ax2.plot([0,0],'k',linewidth = 6,alpha= 0.4,label = leg_lab2[0])
tmp2 = ax2.plot([0,0],'k',linewidth = 6,alpha= 0.4,linestyle = '--',label = leg_lab2[1])
ax2.legend(loc = 2,prop={'size':10, 'family' : 'normal','weight' : 'bold'},ncol = 1).draw_frame(False)
tmp.pop(0).remove()
tmp2.pop(0).remove()


ax2.set_ylim([10,31])
ax2.set_ylabel('atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax2.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)
ax2.set_xlim([0,240])

#plt.arrow(0,0,0,1, shape='full', lw=3, length_includes_head=True, head_width=.01)


a1 = matplotlib.patches.Arrow(0.5-0.01,0.5+0.01,0.05,0.0, width=0.8,edgecolor='none',facecolor='gray',fill=True,transform=fig.transFigure, figure=fig,alpha=0.25)

fig.lines.extend([a1])
fig.canvas.draw()

plt.tight_layout()
plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_1b.png')
plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_1b.pdf')
plt.show(block = False)
#plt.close('all')


'''
spg-stg difference plots
'''


#for i in range(4):
#  ax1.plot(co2[:,i],linewidth = 6,alpha= 0.4)
#
#ax1.set_ylabel('atm. CO$_2$ (ppm)', multialignment='center',fontweight='bold',fontsize = 14)
#ax1.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)

#plt.close('all')


colours = ['b','r']



fig, (ax1) = plt.subplots(1,1,figsize=(5, 4))
ax1.plot(results[:,0]-results[0,0],results[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '-',color=colours[0])
ax2 = ax1.twinx()
ax2.plot(results[:,0]-results[0,0],results_stg[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '--',color=colours[1])

ax1.set_xlim([150,160])
min1 = 22
max1 = 27
min1b = -1
max1b = 4
ax1.set_ylim([min1,max1])
ax2.set_ylim([min1b,max1b])

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
ax1.plot(results[:,0]-results[0,0],results[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '-',color=colours[0])
ax1.plot(results[:,0]-results[0,0],results_stg[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '--',color=colours[1])

ax1.set_xlim([155,165])
#min1 = 100
#max1 = 160
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
3
'''

fig, (ax1) = plt.subplots(1,1,figsize=(5, 4))
ax1.plot(results[:,0]-results[0,0],results[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '-',color=colours[0])
ax1.plot(results[:,0]-results[0,0],results_stg[:,i+1],linewidth = 6,alpha= 0.4,linestyle = '--',color=colours[1])

ax1.set_xlim([170,180])
#min1 = 100
#max1 = 160
ax1.set_ylim([min1,max1])
ax1.set_ylabel('atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_ylabel('Subtropical atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)

plt.tight_layout()
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_2.png')
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/mechanism_2.pdf')
plt.show(block = False)
#plt.close('all')



results = np.genfromtxt('/home/ph290/box_modelling/boxmodel_6_box_back_to_basics/results/rcp85_spg_box_model_qump_results_3.csv',delimiter = ',')
results_stg = np.genfromtxt('/home/ph290/box_modelling/boxmodel_6_box_back_to_basics/results/rcp85_stg_box_model_qump_results_3.csv',delimiter = ',')


mpl.rcdefaults()

font = {'family' : 'monospace',
        'weight' : 'bold',
        'family' : 'serif',
        'size'   : 14}

mpl.rc('font', **font)


plt.close('all')

fig, (ax1) = plt.subplots(1,1,figsize=(5, 4))


for i in range(1):
  ax1.plot(results[:,0]-results[0,0],results[:,i+1],'k',linewidth = 6,alpha= 0.4)

ax1b = ax1.twinx()
for i in range(1):
  ax1b.plot(results[:,0]-results[0,0],results_stg[:,i+1],'k',linewidth = 6,alpha= 0.4,linestyle = '--')


ax1.set_ylabel('spg atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1b.set_ylabel('stg (dasshed) atm. [CO$_2$] minus ocean [CO$_2$]\n(ppm)', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_xlabel('year', multialignment='center',fontweight='bold',fontsize = 14)
ax1.set_xlim([0,240])



plt.tight_layout()
plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/rcp_85.png')
plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/rcp_85.pdf')
plt.show(block = False)
#plt.close('all')
