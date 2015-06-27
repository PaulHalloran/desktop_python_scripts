import numpy as np
import matplotlib.pyplot as plt
import matplotlib

tmp_str=np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'])
dir_name='/home/ph290/box_modelling/co2_files/new/'
dir_name2='/home/ph290/box_modelling/alk_files/new/'

box_output_file='/home/ph290/box_modelling/box_model_mac/results/box_model_qump_results_1.csv'
box_output=np.genfromtxt(box_output_file, delimiter=",")

co2_input=np.copy(box_output)
alk_input=np.copy(box_output)

for i in np.arange(12):
	tmp=np.genfromtxt(dir_name+'co2_'+tmp_str[i]+'.txt', delimiter=",")
	co2_input[:,0]=tmp[:,0]
	co2_input[:,i+1]=tmp[:,1]
	tmp=np.genfromtxt(dir_name2+'alk_'+tmp_str[i]+'.txt', delimiter=",")
	alk_input[:,0]=tmp[:,0]
	alk_input[:,i+1]=tmp[:,1]
	
box_output_file='/home/ph290/box_modelling/box_model_mac/results/box_model_qump_results_1.csv'
box_output=np.genfromtxt(box_output_file, delimiter=",")

linestyles = ['-', '-', '--', ':']
fig=plt.figure()

ax1 = plt.subplot2grid((2,3),(0,0))
ax1.set_title('CO$_2$ conc. scenario')
ax1.set_xlabel('year')
ax1.set_ylabel('conc.')
for i in np.arange(4):
	line,=ax1.plot(co2_input[:,0],co2_input[:,i+1],'k'+linestyles[i])
	if i ==1:
		line.set_dashes([8, 4, 2, 4, 2, 4]) 

ax2 = plt.subplot2grid((2,3),(1,0))
ax2.set_title('alkalinity scenario')
ax2.set_xlabel('year')
ax2.set_ylabel('conc.')
ax2.set_ylim(2220,2320)
for i in np.arange(4):
	line,=ax2.plot(alk_input[:,0],alk_input[:,i+1+4])
	
ax3 = plt.subplot2grid((2,3),(0,1))
ax3.set_title('flux response to CO$_2$ scenario')
ax3.set_xlabel('year')
ax3.set_ylabel('flux')
for i in np.arange(4):
	line,=ax3.plot(box_output[:,0],box_output[:,i+1],'k'+linestyles[i])
	if i ==1:
		line.set_dashes([8, 4, 2, 4, 2, 4]) 
		
ax4 = plt.subplot2grid((2,3),(1,1))
ax4.set_title('flux response to alkalinity scenario')
ax4.set_xlabel('year')
ax4.set_ylabel('flux')
for i in np.arange(4):
	line,=ax4.plot(box_output[:,0],box_output[:,i+4+1])
	
ax5 = plt.subplot2grid((2,3),(0,2), rowspan=2)
ax5.set_title('flux response to CO$_2$ and alkalinity scenarios')
ax5.set_xlabel('year')
ax5.set_ylabel('flux')
for i in np.arange(4):
	line,=ax5.plot(box_output[:,0],box_output[:,i+8+1],linestyles[2])

a1 = matplotlib.patches.Arrow(0.325,0.75,0.1,0.0, width=0.1,edgecolor='k',facecolor='none',fill=False,transform=fig.transFigure, figure=fig)
a2 = matplotlib.patches.Arrow(0.325,0.25,0.1,0.0, width=0.1,edgecolor='k',facecolor='none',fill=False,transform=fig.transFigure, figure=fig)
a3 = matplotlib.patches.Arrow(0.325+0.275,0.75,0.1,0.0, width=0.1,edgecolor='k',facecolor='none',fill=False,transform=fig.transFigure, figure=fig)
a4 = matplotlib.patches.Arrow(0.325+0.275,0.25,0.1,0.0, width=0.1,edgecolor='k',facecolor='none',fill=False,transform=fig.transFigure, figure=fig)

fig.lines.extend([a1, a2,a3,a4])
fig.canvas.draw()

plt.show()
