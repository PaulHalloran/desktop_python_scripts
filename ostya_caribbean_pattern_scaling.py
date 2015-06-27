import iris
import numpy as np
import matplotlib.pyplot as plt
import iris.analysis
import iris.coord_categorisation
import iris.analysis.stats
import scipy.stats
import iris.quickplot as qplt
import iris.coords as icoords
import cartopy.crs as ccrs
import cartopy
import iris.plot as iplt
import matplotlib as mpl

'''

file = '/home/ph290/data1/observations/ostia/METOFFICE-GLO-SST-L4-RAN-OBS-SST-SEAS_1415982499195.nc'

cube = iris.load_cube(file,'sea_surface_temperature')

'''

lon_west = 275
lon_east = 350
lat_south = -20
lat_north = 60.0 

cube_region_tmp = cube.intersection(longitude=(lon_west, lon_east))
cube_caribbean = cube_region_tmp.intersection(latitude=(lat_south, lat_north))
iris.coord_categorisation.add_year(cube_caribbean, 'time', name='year')
atlantic_cube = cube_caribbean.aggregated_by('year', iris.analysis.MEAN)

lon_west = 260
lon_east = 300
lat_south = 0
lat_north = 30.0 

cube_region_tmp = cube.intersection(longitude=(lon_west, lon_east))
cube_caribbean = cube_region_tmp.intersection(latitude=(lat_south, lat_north))
iris.coord_categorisation.add_year(cube_caribbean, 'time', name='year')
cube_caribbean = cube_caribbean.aggregated_by('year', iris.analysis.MEAN)

lon_west1 = 300
lon_east1 = 330
lat_south1 = 5
lat_north1 = 20.0 

cube_region_tmp = cube.intersection(longitude=(lon_west1, lon_east1))
cube_trop_atl = cube_region_tmp.intersection(latitude=(lat_south1, lat_north1))
cube_trop_atl.coord('latitude').guess_bounds()
cube_trop_atl.coord('longitude').guess_bounds()
iris.coord_categorisation.add_year(cube_trop_atl, 'time', name='year')
cube_trop_atl1 = cube_trop_atl.aggregated_by('year', iris.analysis.MEAN)

lon_west2 = 280
lon_east2 = 315
lat_south2 = 25
lat_north2 = 45.0 

cube_region_tmp = cube.intersection(longitude=(lon_west2, lon_east2))
cube_trop_atl = cube_region_tmp.intersection(latitude=(lat_south2, lat_north2))
cube_trop_atl.coord('latitude').guess_bounds()
cube_trop_atl.coord('longitude').guess_bounds()
iris.coord_categorisation.add_year(cube_trop_atl, 'time', name='year')
cube_trop_atl2 = cube_trop_atl.aggregated_by('year', iris.analysis.MEAN)

cube_trop_atl_mean1 = cube_trop_atl1.collapsed(['latitude','longitude'],iris.analysis.MEAN)
cube_trop_atl_mean2 = cube_trop_atl2.collapsed(['latitude','longitude'],iris.analysis.MEAN)

tmp_cube1 = cube_caribbean.copy()
tmp_cube2 = cube_caribbean.copy()
data1 = tmp_cube1.data.copy()
data2 = tmp_cube2.data.copy()

size = np.shape(atlantic_cube.data)

for i in range(size[0]):
    data1.data[i] = data1.data[i] * 0.0 + cube_trop_atl_mean1.data[i]
    data2.data[i] = data2.data[i] * 0.0 + cube_trop_atl_mean2.data[i]

tmp_cube1.data = data1
tmp_cube2.data = data2

a = cube_caribbean.data
b1 = tmp_cube1.data
b2 = tmp_cube2.data
out1 = a[0].copy()
out2 = a[0].copy()
out3 = a[0].copy()

shape = a.shape
for i in range(shape[1]):
	for j in range(shape[2]):
		out1[i,j] = scipy.stats.pearsonr(a[:,i,j], b1[:,i,j])[0]
		out2[i,j] = scipy.stats.pearsonr(a[:,i,j], b2[:,i,j])[0]
		
for i in range(shape[1]):
	for j in range(shape[2]):	
		x = np.array([b1[:,i,j],b2[:,i,j]])
		y = np.array(a[:,i,j])
		n = np.max(x.shape)    
		X = np.vstack([np.ones(n), x]).T
		result = np.linalg.lstsq(X, y)[0]
		prediction = result[0] + result[1]*b1[:,i,j] + result[2]*b2[:,i,j]
		out3[i,j] = scipy.stats.pearsonr(y, prediction)[0]

out_cube1 = cube_caribbean[0].copy()
out_cube1.data = out1
out_cube2 = cube_caribbean[0].copy()
out_cube2.data = out2
out_cube3 = cube_caribbean[0].copy()
out_cube3.data = out3


levels = np.linspace(-1,1,51)

plt.close('all')
fig = plt.figure()

data=out_cube1.data
lons = out_cube1.coord('longitude').points
lats = out_cube1.coord('latitude').points
cube_label = 'latitude: %s' % out_cube1.coord('latitude').points
# ax1 = fig.add_subplot(211, projection = ccrs.PlateCarree())
ax1 = plt.subplot2grid((2,3), (1,1), colspan=1, projection = ccrs.PlateCarree())
contour = plt.contourf(lons, lats, data,levels=levels,cmap='jet',xlabel=cube_label)
cartopy.feature.LAND.scale='50m'
ax1.add_feature(cartopy.feature.LAND)
ax1.add_feature(cartopy.feature.RIVERS)
ax1.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax1.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax1.coastlines(resolution='50m')

data=out_cube2.data
lons = out_cube2.coord('longitude').points
lats = out_cube2.coord('latitude').points
cube_label = 'latitude: %s' % out_cube2.coord('latitude').points
ax2 = plt.subplot2grid((2,3), (0,1), colspan=1, projection = ccrs.PlateCarree())
contour = plt.contourf(lons, lats, data,levels=levels,cmap='jet',xlabel=cube_label)
cartopy.feature.LAND.scale='50m'
ax2.add_feature(cartopy.feature.LAND)
ax2.add_feature(cartopy.feature.RIVERS)
ax2.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax2.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax2.coastlines(resolution='50m')

data=out_cube3.data
lons = out_cube2.coord('longitude').points
lats = out_cube2.coord('latitude').points
cube_label = 'latitude: %s' % out_cube2.coord('latitude').points
ax3 = plt.subplot2grid((2,3), (0,2), rowspan=2, projection = ccrs.PlateCarree())
contour = plt.contourf(lons, lats, data,levels=levels,cmap='jet',xlabel=cube_label)
cartopy.feature.LAND.scale='50m'
ax3.add_feature(cartopy.feature.LAND)
ax3.add_feature(cartopy.feature.RIVERS)
ax3.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax3.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax3.coastlines(resolution='50m')

ax4 = fig.add_axes([0.45, 0.050, 0.4, 0.08])
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax4,norm=norm,orientation='horizontal')
cb1.set_label('Some Units')

x1 = [lon_west1,lon_east1,lon_east1,lon_west1,lon_west1] 
x2 = [lon_west2,lon_east2,lon_east2,lon_west2,lon_west2]
y1 = [lat_south1,lat_south1,lat_north1,lat_north1,lat_south1]
y2 = [lat_south2,lat_south2,lat_north2,lat_north2,lat_south2]
data=atlantic_cube[0].data*0.0
lons = atlantic_cube[0].coord('longitude').points
lats = atlantic_cube[0].coord('latitude').points
cube_label = 'latitude: %s' % out_cube2.coord('latitude').points
ax5 = plt.subplot2grid((2,3), (0,0), rowspan=2, projection = ccrs.PlateCarree())
contour = plt.contourf(lons, lats, data,levels=levels,cmap='jet',xlabel=cube_label)
cartopy.feature.LAND.scale='50m'
ax5.add_feature(cartopy.feature.LAND)
ax5.add_feature(cartopy.feature.RIVERS)
ax5.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax5.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax5.coastlines(resolution='50m')
ax5.plot(x1, y1, transform=ccrs.PlateCarree())
ax5.fill(x1, y1, color='red', transform=ccrs.PlateCarree(), alpha=0.4)
ax5.plot(x2, y2, transform=ccrs.PlateCarree())
ax5.fill(x2, y2, color='blue', transform=ccrs.PlateCarree(), alpha=0.4)


# plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/carib_pat_scal.ps')
