import carbchem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcdefaults()

fsize = 20

font = {'family' : 'monospace',
        'weight' : 'bold',
        'family' : 'serif',
        'size'   : fsize}

mpl.rc('font', **font)


def alk_field(alk_in,alk_drawdown,array_size,spacing):
    tmp = np.zeros([array_size[0],array_size[1]]) + alk_in
    tmp_1 = np.arange(tmp[:,0].size)
    counter = 0
    for i in tmp_1[0:tmp[:,0].size:spacing]:
        for j in tmp_1[0:tmp[0,:].size:spacing]:
            counter += 1
    value = alk_in-((alk_drawdown*(array_size[0]*array_size[1]))/counter)
    for i in tmp_1[0:tmp[:,0].size:spacing]:
        for j in tmp_1[0:tmp[0,:].size:spacing]:
            tmp[i,j] = value
    print 'point alk value = '+np.str(value)
    print 'mean alkalinity = '+np.str(np.mean(tmp))
    return tmp

t1 = np.zeros(30) + 15.0
s1 = np.zeros(30) + 35.0
#alk = np.zeros(20) + 0.0023
alk1 = np.arange(30)/5.0e4 + 0.0018
#dic = np.arange(20)/5.0e4 + 0.0020
dic1 = np.zeros(30) + 0.0020

alk_in = 0.0023
alk_drawdown = 0.00005
array_size = [14,14]



alk_field1 = alk_field(alk_in,alk_drawdown,array_size,1)
alk_field2 = alk_field(alk_in,alk_drawdown,array_size,2)
alk_field3 = alk_field(alk_in,alk_drawdown,array_size,3)

t = np.zeros([array_size[0],array_size[1]]) + t1[0]
s = np.zeros([array_size[0],array_size[1]]) + s1[0]
dic = np.zeros([array_size[0],array_size[1]]) + dic1[0]

vmax = 0.0024
vmin = 0.0018

plt.close('all')
plt.figure()
plt.pcolor(alk_field1,cmap='jet',vmin = vmin,vmax = vmax)
plt.colorbar()
plt.title('mean alkalinity = '+np.str(np.mean(alk_field1))+' (mol kg$^{-1}$)')
plt.savefig('/home/ph290/Documents/figures/alkalinity_nonlinearity/alk_1.pdf',transparent = True)
#plt.show(block = False)

plt.figure()
plt.pcolor(alk_field2,cmap='jet',vmin = vmin,vmax = vmax)
plt.colorbar()
plt.title('mean alkalinity = '+np.str(np.mean(alk_field1))+' (mol kg$^{-1}$)')
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/alkalinity_nonlinearity/alk_2.pdf',transparent = True)

plt.figure()
plt.pcolor(alk_field3,cmap='jet',vmin = vmin,vmax = vmax)
plt.colorbar()
plt.title('mean alkalinity = '+np.str(np.mean(alk_field1))+' (mol kg$^{-1}$)')
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/alkalinity_nonlinearity/alk_3.pdf',transparent = True)


co2_1 = carbchem.carbchem(1,-99.9,t,s,dic,alk_field1)
co2_2 = carbchem.carbchem(1,-99.9,t,s,dic,alk_field2)
co2_3 = carbchem.carbchem(1,-99.9,t,s,dic,alk_field3)


vmin1 = 100
vmax1 = 3000

plt.figure()
plt.pcolor(co2_1,cmap='jet',vmin = vmin1,vmax = vmax1)
plt.colorbar()
plt.title('mean pCO$_2$ = '+np.str(np.mean(co2_1))+' (uatm.)')
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/alkalinity_nonlinearity/co2_1.pdf',transparent = True)

plt.figure()
plt.pcolor(co2_2,cmap='jet',vmin = vmin1,vmax = vmax1)
plt.colorbar()
plt.title('mean pCO$_2$ = '+np.str(np.mean(co2_2))+' (uatm.)')
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/alkalinity_nonlinearity/co2_2.pdf',transparent = True)

plt.figure()
plt.pcolor(co2_3,cmap='jet',vmin = vmin1,vmax = vmax1)
plt.colorbar()
plt.title('mean pCO$_2$ = '+np.str(np.mean(co2_3))+' (uatm.)')
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/alkalinity_nonlinearity/co2_3.pdf',transparent = True)

plt.close('all')


min_alk = [np.min(alk_field1),np.min(alk_field2),np.min(alk_field3)]
max_alk = [np.max(alk_field1),np.max(alk_field2),np.max(alk_field3)]

min_co2 = [np.min(co2_1),np.min(co2_2),np.min(co2_3)]
max_co2 = [np.max(co2_1),np.max(co2_2),np.max(co2_3)]

x =  carbchem.carbchem(1,-99.9,t1,s1,dic1,alk1)

plt.close('all')
plt.figure
plt.plot(alk,x,linewidth = 3)

colours = ['r','g','black']
s=[400,400,400]
m = ['D','o','*']

for i,tmp in enumerate(min_alk):
    plt.scatter(min_alk[i],max_co2[i],color = colours[i],marker=m[i],s = s[i])
    plt.scatter(max_alk[i],min_co2[i],color = colours[i],marker=m[i],s = s[i])

plt.xlabel('alkalinity (mol kg$^{-1}$)',fontweight='bold',fontsize = fsize)
plt.ylabel('pCO$_2$ (uatm.)',fontweight='bold',fontsize = fsize)
plt.xlim([0.00185,0.00235])
plt.ylim([-1000,4000.0])
plt.tight_layout()

#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/alkalinity_nonlinearity/alk_line.pdf',transparent = True)

