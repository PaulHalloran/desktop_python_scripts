import numpy as np
import scipy
import matplotlib.pyplot as plt

file='/home/ph290/data1/temp_data/misc_data/data_from_wanamaker_2012.csv'

data=np.genfromtxt(file,skip_header=1,delimiter=",",usecols=[0,1,2,3])

l_wdth=2

fig = plt.figure(num=None, figsize=(7, 3), dpi=150, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
ax.set_yscale('log')
ln1=ax.plot(data[:,0],data[:,1], 'b-', linewidth=l_wdth,label='Bivalve a')
ln2=ax.plot(data[:,2],data[:,3],'r-', linewidth=l_wdth/2,label='Bivalve b')
plt.ylim(0,1000)
plt.xlim(1500,2010)
plt.xlabel('year')
plt.ylabel('increment width (logged)')
lns=ln1+ln2
labs = [l.get_label() for l in lns]
#plt.legend(lns, labs).draw_frame(False)
plt.legend(lns, labs,prop={'size':tmp_font_size}).draw_frame(False)

for item in ([ax.xaxis.label, ax.yaxis.label]):
	item.set_fontsize(tmp_font_size)
	item.set_weight('bold')

# plt.show()
plt.savefig('/home/ph290/Desktop/delete4.pdf')

