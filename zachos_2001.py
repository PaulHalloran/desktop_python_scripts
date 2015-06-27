import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pylab import *

file = '/home/ph290/Documents/teaching/data$ vim zachos2001.txt'
data = np.genfromtxt('zachos2001.txt',skip_header = 90,usecols = (1, 3))

colour = 'white'
colour2 = 'red'

rc('axes', linewidth=2)

def plot_my_fig(data,xtitle,ytitle,colour,colour2,xrang,yrang,name):
    fsize = 16
    fontsize = 14
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[:,0],data[:,1],color = colour2,linewidth= 2, alpha=0.5)
    ax.scatter(data[:,0],data[:,1],facecolors='none', edgecolors=colour2, alpha=0.5)
    ax.set_xlim(xrang)
    ax.set_ylim(yrang)
    ax.set_ylabel(xtitle, fontsize=fsize,fontweight='bold')
    ax.set_xlabel(ytitle, fontsize=fsize,fontweight='bold')
    ax.spines['bottom'].set_color(colour)
    ax.spines['top'].set_color(colour)
    ax.spines['left'].set_color(colour)
    ax.spines['right'].set_color(colour)
    ax.xaxis.label.set_color(colour)
    ax.yaxis.label.set_color(colour)
    ax.tick_params(axis='x', colors=colour)
    ax.tick_params(axis='y', colors=colour)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    #plt.show()
    plt.savefig('/home/ph290/Documents/figures/'+name, transparent=True,dpi = 500)


plot_my_fig(data,'d$^{18}$O','Time (millions of years before present)',colour,colour2,[0,70],[6,-2],'zachos_1.png')

plot_my_fig(data,'d$^{18}$O','Time (millions of years before present)',colour,colour2,[50,60],[2,-2],'zachos_2.png')

plot_my_fig(data,'d$^{18}$O','Time (millions of years before present)',colour,colour2,[54,56],[2,-2],'zachos_3.png')

plot_my_fig(data,'d$^{18}$O','Time (millions of years before present)',colour,colour2,[0,5],[6,2],'zachos_4.png')

plot_my_fig(data,'d$^{18}$O','Time (millions of years before present)',colour,colour2,[0,0.5],[6,2],'zachos_5.png')

