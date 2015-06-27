'''

import numpy as np
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import iris.coord_categorisation
import iris.analysis
import running_mean

#file = '/home/ph290/data1/observations/hadcrut4/HadCRUT.4.2.0.0.median.nc'
#cube = iris.load_cube(file,'near_surface_temperature_anomaly')

file2 = '/home/ph290/data1/observations/hadisst/HadISST_sst.nc'
cube = iris.load_cube(file2)

iris.coord_categorisation.add_year(cube, 'time', name='year')
cube = cube.aggregated_by('year', iris.analysis.MEAN)


#qplt.contourf(cube[-1])
#plt.gca().coastlines()
#plt.show()


cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(cube)

cube2 = cube.copy()
cube3 = cube.copy()

x, y = iris.analysis.cartography.get_xy_grids(cube2)


lon_west = -180
lon_east = 180
lat_south = -90
lat_north = 0
loc = np.where((x >= lon_west) & (x <= lon_east) & (y >= lat_south) & (y <= lat_north))

cube2.data[:,loc[0],loc[1]] = np.nan
cube2.data.mask[:,loc[0],loc[1]] = True
#qplt.contourf(cube2[-1])
#plt.gca().coastlines()
#plt.show()

lon_west = -100
lon_east = 20
lat_south = 0.0
lat_north = 75

lon_west = -180
lon_east = 180
lat_south = 0
lat_north = 90

loc2 = np.where((x >= lon_west) & (x <= lon_east) & (y >= lat_south) & (y <= lat_north))

cube3.data[:,loc2[0],loc2[1]] = np.nan
cube3.data.mask[:,loc2[0],loc2[1]] = True
qplt.contourf(cube3[-1])
plt.gca().coastlines()
plt.show()

ts0 = cube.collapsed(['latitude','longitude'], iris.analysis.MEAN, weights=grid_areas)
ts1 = cube2.collapsed(['latitude','longitude'], iris.analysis.MEAN, weights=grid_areas)
ts2 = cube3.collapsed(['latitude','longitude'], iris.analysis.MEAN, weights=grid_areas)

coord = ts0.coord('time')
dt = coord.units.num2date(coord.points)
year = np.array([coord.units.num2date(value).year for value in coord.points])

'''

file_forcing = '/home/ph290/data1/cmip5/forcing_data/volcanic_forcing.txt'
data_volc = np.genfromtxt(file_forcing,skip_header = 6)

file_forcing = '/home/ph290/data1/cmip5/forcing_data/rcp_forcing_data/historical_and_rcp85_atm_co2.txt'
data_co2 = np.genfromtxt(file_forcing,skip_header = 1,delimiter= ',')

meaning = 10
lnwdth = 4

fig, ax1 = plt.subplots(figsize=(6, 12))
ax1.plot(year,running_mean.running_mean(ts0.data-ts0[0].data,meaning),'b',linewidth = lnwdth,alpha = 0.75)
ax1.plot(year,running_mean.running_mean(ts1.data-ts1[0].data,meaning),'r',linewidth = lnwdth,alpha = 0.75)
ax1.plot(year,running_mean.running_mean(ts2.data-ts2[0].data,meaning),'g',linewidth = lnwdth,alpha = 0.75)

ax2 = ax1.twinx()
#ax2.plot(data[:,0],data[:,1])
ax2.plot(data_volc[:,0],data_volc[:,2],'k--',linewidth = lnwdth,alpha = 0.5) # N
ax2.plot(data_volc[:,0],data_volc[:,3],'k-',linewidth = lnwdth,alpha = 0.5) # S

ax3 = ax2.twinx()
loc = np.where((data_co2[:,0] >= np.min(year)) & (data_co2[:,0] <= np.max(year)))[0]
ax3.plot(data_co2[loc[0]:loc[-1],0],data_co2[loc[0]:loc[-1],1],'k',linewidth = lnwdth,alpha = 0.75)

ax1.set_xlim([1860,2010])
ax1.set_ylim([-0.5,1.2])
ax2.set_ylim([0.0,1])
ax3.set_ylim([150,400])

#plt.show()
plt.savefig('/home/ph290/Documents/figures/amo_justificatoin.pdf')


loc = np.where((year >= 1935) & (year <= 1955))[0]
loc2 = np.where((year >= 1960) & (year <= 1980))[0]

cube_coll1 = cube[loc].collapsed('time',iris.analysis.MEAN)
cube_coll2 = cube[loc2].collapsed('time',iris.analysis.MEAN)

plt.figure()
qplt.contourf(cube_coll2-cube_coll1,50)
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/amo_justification_map.pdf',transparent = True)

