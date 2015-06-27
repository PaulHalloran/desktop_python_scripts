import numpy
from matplotlib.pyplot import *
from iris import *
from iris.analysis import *
import iris.quickplot as quickplot
import seawater
from matplotlib.ticker import ScalarFormatter
import iris.plot as iplt

directory = '/home/ph290/Documents/teaching/'

salinity_file = 'salinity_annual_1deg.nc'
temperature_file = 'temperature_annual_1deg.nc'

salinity_cube = load(directory+salinity_file,'Statistical Mean')[0][0]
temperature_cube = load(directory+temperature_file,'Statistical Mean')[0][0]


density_cube = temperature_cube.copy()
density_cube.standard_name = 'sea_water_density'
density_cube.units = 'kg m-3'

density_cube.data = seawater.dens(salinity_cube.data,temperature_cube.data,1)
density_cube_meridional = density_cube.collapsed('longitude',MEAN)

maps=[m for m in cm.datad if not m.endswith("_r")]

#figure()
#quickplot.contourf(density_cube_meridional, 30,cmap=get_cmap(maps[14])) 
#CS = quickplot.contour(density_cube_meridional,[1028.0,1027.9,1027.75,1027.5,1027.0,1026.5,1026.0],colors='k',linewideths=30)
#clabel(CS, fontsize=9, inline=1)
#savefig('/home/ph290/Documents/figures/isopycnals.pdf')

fig = figure()
ax = iplt.contourf(density_cube_meridional, numpy.linspace(1026.0,1028,50),cmap=get_cmap(maps[14]))
cbar = colorbar(ax)
cbar.formatter.set_useOffset(False)
cbar.set_label('kg m$^{-3}$')

xlim(-80,-40)
ylim(3000,0)
CS = iplt.contour(density_cube_meridional,[1028.0,1027.9,1027.75,1027.5,1027.0,1026.5,1026.0],colors='k',linewideths=30)
clabel(CS, fontsize=9, inline=1)

#show()
savefig('/home/ph290/Documents/figures/isopycnals_b.pdf',transparent = True)


#plt.show()
