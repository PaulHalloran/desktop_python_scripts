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


lon_west = -170.0
lon_east = -120
lat_south = -5
lat_north = 5

region = iris.Constraint(longitude=lambda v: lon_west <= v <= lon_east,latitude=lambda v: lat_south <= v <= lat_north)
cube_region = cube4.extract(region)

cube_region.coord('longitude').guess_bounds()
cube_region.coord('latitude').guess_bounds()
grid_areas_region = iris.analysis.cartography.area_weights(cube_region)



cube_region_detrended = cube_region.copy()
	
cube_region_detrended.data = scipy.signal.detrend(cube_region.data)

coord = cube_region_detrended.coord('time')
year = np.array([coord.units.num2date(value).year for value in coord.points])
month = np.array([coord.units.num2date(value).month for value in coord.points])

cube_region_detrended_mean = cube_region_detrended.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas_region)

import iris.quickplot as qplt

mpl.rcParams['figure.figsize'] = 10, 5
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 20



data = cube_region_detrended_mean.data
coord = cube_region_detrended_mean.coord('time')
dt = coord.units.num2date(coord.points)
year = np.array([coord.units.num2date(value).year for value in coord.points])
month  = np.array([coord.units.num2date(value).month for value in coord.points])

fig, (ax)  = plt.subplots(nrows=1)
ax.plot(year+month/12.0,data,'r',linewidth = 3)
ax.set_xlabel('year')
ax.set_ylabel('ENSO box SST anomaly ($^o$C)')
ax.set_xlim(1992,2003) 
ax.set_ylim(-0.0000005,0.0000005) 
set_foregroundcolor(ax, 'black')
set_backgroundcolor(ax, 'white')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
mpl.pyplot.locator_params(nbins=4)
plt.tight_layout()
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
#plt.show()                      
plt.savefig('/home/ph290/Documents/figures/enso_ts_b.pdf', transparent=True,dpi = 500)
