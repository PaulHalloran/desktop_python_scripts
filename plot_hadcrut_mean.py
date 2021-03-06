import numpy as np
import matplotlib.pyplot as plt
import iris
import iris.analysis
import iris.quickplot as qplt
import iris.analysis.cartography
import cartopy.crs as ccrs

colour = 'black'
colour2 = 'red'

def plot_my_fig(date,data,xtitle,ytitle,colour,colour2,xrang,yrang,name):
    fsize = 16
    fontsize = 14
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(date,data,color = colour2,linewidth= 2, alpha=0.9)
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
    plt.savefig('/home/ph290/Documents/figures/'+name+'.png', transparent=True,dpi = 500)
    plt.close()

t_file = '/home/ph290/data1/observations/hadcrut4/HadCRUT.4.2.0.0.median.nc'

cube = iris.load_cube(t_file,'near_surface_temperature_anomaly')
cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(cube)
cube_mean = cube.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas)

coord = cube.coord('time')
year = np.array([coord.units.num2date(value).year for value in coord.points])
month = np.array([coord.units.num2date(value).month for value in coord.points])

plot_my_fig(year+month/12.0,cube_mean.data,'global avg. surafce temperature anomaly','year',colour,colour2,[1850,2010],[-1,1],'hadcrut')

cube_mean2 = cube[-20*12.0:-1].collapsed('time', iris.analysis.MEAN)
#cube_mean2b = cube[0:20*12.0].collapsed('time', iris.analysis.MEAN)
#cube_mean2 = cube_mean2a-cube_mean2b
qplt.contourf(cube_mean2,51)
plt.gca().coastlines()
#plt.show()
plt.savefig('/home/ph290/Documents/figures/hadcrut4_last20yrs.png', transparent=True,dpi = 500)
