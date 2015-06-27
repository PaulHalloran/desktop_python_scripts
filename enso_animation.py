import iris
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import iris.coord_categorisation
import iris.analysis.cartography
import iris.analysis
import scipy
from scipy import signal
import numpy as np
import cartopy

file = '/data/data1/ph290/observations/hadisst/HadISST_sst.nc'
cube = iris.load(file)
cube = cube[0]
iris.coord_categorisation.add_year(cube, 'time', name='year')
iris.coord_categorisation.add_month(cube, 'time', name='month')
cube2 = cube.aggregated_by('month', iris.analysis.MEAN)

months = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

cube3 = cube.data.copy()
cube4 = cube.copy()
for i in range(cube.shape[0]):
	month_tmp = cube[i].coord('month').points[0]
	loc = np.where(months == month_tmp)[0][0]
	cube3[i,:,:] -= cube2[loc].data

cube4.data = cube3
	
print 'extract lats etc.'
plt_cube = cube4
lats = plt_cube[0].coord('latitude').points
lons = plt_cube[0].coord('longitude').points
lons, lats = np.meshgrid(lons, lats)

print 'generate filenames'
import itertools
letters = ['a','b','c','d','e']
x = list(itertools.product(letters, repeat = np.size(letters)))

coord = cube.coord('time')
year = np.array([coord.units.num2date(value).year for value in coord.points])
month = np.array([coord.units.num2date(value).month for value in coord.points])

#for i in np.arange(cube4.shape[0]):
for i in range(1689,1720):
	print np.str(i)+' of '+(np.str(cube.shape[0]))
	plt.figure()
	ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=200.0))
#ccrs.Orthographic(central_longitude=200.0, central_latitude=0.0))
	#ccrs.Orthographic(central_longitude=220.0, central_latitude=0.0))
	ax.set_global()
	CS = ax.contourf(lons,lats,cube4[i].data,np.linspace(-3.0,3.0,51),transform = ccrs.PlateCarree())
	ax.add_feature(cartopy.feature.LAND)
	ax.coastlines()
	ax.set_title(months[month[i]-1]+' '+np.str(year[i]))
	cbar = plt.colorbar(CS)
	cbar.ax.set_xlabel('SST $^o$C')
	#plt.show()
	plt.savefig('/home/ph290/Documents/figures/anim3/enso'+(''.join(x[i]))+'.png',dpi = 100)
