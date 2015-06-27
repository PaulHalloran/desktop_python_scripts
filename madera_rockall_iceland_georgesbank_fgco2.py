import iris
import glob
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
import scipy.ndimage.filters
import scipy
from scipy import signal
import scipy.stats
import time


directory = '/media/usb_external1/cmip5/fgco2_regridded_hist'

files = glob.glob(directory+'/*fgco2_*_regridded.nc')

models = []
for file in files:
    models.append(file.split('/')[-1].split('_')[0])

models = np.unique(np.array(models))

out_data1 = np.zeros([np.size(models),180,360])
out_data1[:] = np.nan
out_data2 = np.copy(out_data1)
out_data3 = np.copy(out_data1)
out_data4 = np.copy(out_data1)
out_data5 = np.copy(out_data1)
out_data6 = np.copy(out_data1)
	
for mod_num,model in enumerate(models):
	print model
	cube = iris.load_cube(directory+'/'+model+'_*fgco2*_regridded.nc')
	try:
		iris.coord_categorisation.add_year(cube, 'time', name='year')
	except:
		print 'alreday has year coordinate'
	cube = cube.aggregated_by('year', iris.analysis.MEAN)
	cube.data = scipy.signal.detrend(cube.data, axis=0)

	lon_west = -14
	lon_east = -13
	lat_south = 57
	lat_north = 58

	try:
    		cube.coord('latitude').guess_bounds()
	except:
    		print 'cube already has latitude bounds' 
	try:
		cube.coord('longitude').guess_bounds()
	except:
	    print 'cube already has longitude bounds'

	grid_areas = iris.analysis.cartography.area_weights(cube)
	area_avged_cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

	cube = cube - area_avged_cube

	cube_region = cube.intersection(longitude=(lon_west, lon_east))
	cube_rockall = cube_region.intersection(latitude=(lat_south, lat_north))
	cube_rockall_mean = cube_rockall.collapsed(['latitude','longitude'],iris.analysis.MEAN)

	lon_west = -17
	lon_east = -16
	lat_south = 32
	lat_north = 33

	cube_region = cube.intersection(longitude=(lon_west, lon_east))
	cube_madeira = cube_region.intersection(latitude=(lat_south, lat_north))
	cube_madeira_mean = cube_madeira.collapsed(['latitude','longitude'],iris.analysis.MEAN)
	
	lon_west = -65
	lon_east = -64
	lat_south = 32
	lat_north = 33

	cube_region = cube.intersection(longitude=(lon_west, lon_east))
	cube_bermuda = cube_region.intersection(latitude=(lat_south, lat_north))
	cube_bermuda_mean = cube_bermuda.collapsed(['latitude','longitude'],iris.analysis.MEAN)
	
	lon_west = -67
	lon_east = -66
	lat_south = 41
	lat_north = 42

	cube_region = cube.intersection(longitude=(lon_west, lon_east))
	cube_georgesbank = cube_region.intersection(latitude=(lat_south, lat_north))
	cube_georgesbank_mean = cube_georgesbank.collapsed(['latitude','longitude'],iris.analysis.MEAN)

	lon_west = -24
	lon_east = -23
	lat_south = 15
	lat_north = 16

	cube_region = cube.intersection(longitude=(lon_west, lon_east))
	cube_cape_verde = cube_region.intersection(latitude=(lat_south, lat_north))
	cube_cape_verde_mean = cube_cape_verde.collapsed(['latitude','longitude'],iris.analysis.MEAN)

        lon_west = -20 
        lon_east = -19
        lat_south = 62
        lat_north = 63

        cube_region = cube.intersection(longitude=(lon_west, lon_east))
        cube_s_iceland = cube_region.intersection(latitude=(lat_south, lat_north))
        cube_s_iceland_mean = cube_s_iceland.collapsed(['latitude','longitude'],iris.analysis.MEAN)

	a = cube.data
	b1 = cube_rockall_mean.data
	b2 = cube_madeira_mean.data
	b3 = cube_bermuda_mean.data
	b4 = cube_georgesbank_mean.data
	b5 = cube_cape_verde_mean.data
	b6 = cube_s_iceland_mean.data
	out1 = a[0].copy()
	out2 = a[0].copy()
	out3 = a[0].copy()
	out4 = a[0].copy()
	out5 = a[0].copy()
        out6 = a[0].copy()

	smoothing_years = 10
	b1_tmp = scipy.ndimage.filters.gaussian_filter(b1, smoothing_years)
	b2_tmp = scipy.ndimage.filters.gaussian_filter(b2, smoothing_years)
	b3_tmp = scipy.ndimage.filters.gaussian_filter(b3, smoothing_years)
	b4_tmp = scipy.ndimage.filters.gaussian_filter(b4, smoothing_years)
	b5_tmp = scipy.ndimage.filters.gaussian_filter(b5, smoothing_years)
        b6_tmp = scipy.ndimage.filters.gaussian_filter(b6, smoothing_years)


 	#b1_tmp = b1
 	#b2_tmp = b2
 	#b3_tmp = b3
 	#b4_tmp = b4
 	#b5_tmp = b5
        #b6_tmp = b6
	
	shape = a.shape
	for i in range(shape[1]):
			for j in range(shape[2]):
				a_tmp = scipy.ndimage.filters.gaussian_filter(a[:,i,j], smoothing_years)
 				#a_tmp = a[:,i,j]
				out1[i,j] = scipy.stats.pearsonr(a_tmp, b1_tmp)[0]
				out2[i,j] = scipy.stats.pearsonr(a_tmp, b2_tmp)[0]
				out3[i,j] = scipy.stats.pearsonr(a_tmp, b3_tmp)[0]
				out4[i,j] = scipy.stats.pearsonr(a_tmp, b4_tmp)[0]
				out5[i,j] = scipy.stats.pearsonr(a_tmp, b5_tmp)[0]
                                out6[i,j] = scipy.stats.pearsonr(a_tmp, b6_tmp)[0]

	out_data1[mod_num,:,:] = out1
	out_data2[mod_num,:,:] = out2
	out_data3[mod_num,:,:] = out3
	out_data4[mod_num,:,:] = out4
	out_data5[mod_num,:,:] = out5
        out_data6[mod_num,:,:] = out6



out_data1_mean = scipy.stats.nanmean(out_data1,axis = 0)
out_cube1 = cube[0].copy()
out_cube1.data = out_data1_mean

out_data2_mean = scipy.stats.nanmean(out_data2,axis = 0)
out_cube2 = cube[0].copy()
out_cube2.data = out_data2_mean

out_data3_mean = scipy.stats.nanmean(out_data3,axis = 0)
out_cube3 = cube[0].copy()
out_cube3.data = out_data3_mean

out_data4_mean = scipy.stats.nanmean(out_data4,axis = 0)
out_cube4 = cube[0].copy()
out_cube4.data = out_data4_mean

out_data5_mean = scipy.stats.nanmean(out_data5,axis = 0)
out_cube5 = cube[0].copy()
out_cube5.data = out_data5_mean

out_data6_mean = scipy.stats.nanmean(out_data6,axis = 0)
out_cube6 = cube[0].copy()
out_cube6.data = out_data6_mean

out_data1_std = scipy.stats.nanstd(out_data1,axis = 0)
out_data2_std = scipy.stats.nanstd(out_data2,axis = 0)
out_data3_std = scipy.stats.nanstd(out_data3,axis = 0)
out_data4_std = scipy.stats.nanstd(out_data4,axis = 0)
out_data5_std = scipy.stats.nanstd(out_data5,axis = 0)
out_data6_std = scipy.stats.nanstd(out_data6,axis = 0)

out_cube1_std = cube[0].copy()
out_cube2_std = cube[0].copy()
out_cube3_std = cube[0].copy()
out_cube4_std = cube[0].copy()
out_cube5_std = cube[0].copy()
out_cube6_std = cube[0].copy()

out_cube1_std.data = out_data1_std
out_cube2_std.data = out_data2_std
out_cube3_std.data = out_data3_std
out_cube4_std.data = out_data4_std
out_cube5_std.data = out_data5_std
out_cube6_std.data = out_data6_std

# qplt.contourf(out_cube1,31)
# plt.show()
# 
# qplt.contourf(out_cube2,31)
# plt.show()
# 
# qplt.contourf(out_cube3,31)
# plt.show()

cube_combined = out_cube1 + out_cube2 + out_cube3 + out_cube4 + out_cube6
'''

plt.close('all')
cube_combined.data[np.where(cube_combined.data >= 1)] = 1
cube_combined.data[np.where(cube_combined.data <= -1)] = -1 
qplt.contourf(cube_combined,np.linspace(-1,1,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/airseaflux_correlations_II.png')
#plt.show()

plt.close('all')
fig = plt.figure()
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-30,central_latitude=-45))
ax.set_global()
iplt.contourf(cube_combined,np.linspace(-1,1))
ax.add_feature(cartopy.feature.LAND,facecolor='#f6f6f6')
ax.coastlines(resolution='50m', color='black', linewidth=1)
plt.savefig('/home/ph290/Documents/figures/airseaflux_correlations_III.ps')

'''
lon_west = -90.0
lon_east = 20
lat_south = 0.0
lat_north = 90.0 

cube_combined.data[np.where(cube_combined.data >= 1)] = 1
cube_combined.data[np.where(cube_combined.data <= -1)] = -1

cube_region_tmp = cube_combined.intersection(longitude=(lon_west, lon_east))
cube_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))


lon = [-14.5,-17.5,-65.5,-67.5,-20.5]
lat = [57.5,32.5,32.5,41.5,62.5]

cubes = [out_cube1, out_cube2, out_cube3, out_cube4, out_cube6]


for i,dummy in enumerate(lon):
	cube2 = cubes[i]
	cube2.data[np.where(cube2.data >= 1)] = 1
	cube2.data[np.where(cube2.data <= -1)] = -1
	cube_region_tmp = cube2.intersection(longitude=(lon_west, lon_east))
	cube_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))
	plt.close('all')
	ax = plt.axes(projection=ccrs.PlateCarree())
	#ax.set_global()
	CS = iplt.contourf(cube_region,np.linspace(-1,1,31))
	ax.add_feature(cartopy.feature.LAND,facecolor='#f6f6f6')
	ax.coastlines(resolution='50m', color='black', linewidth=1)
	plt.scatter(lon[i],lat[i],s=300,c = 'k', marker = '*')
	cbar = plt.colorbar(CS)
	#plt.show(block = False)
	plt.savefig('/home/ph290/Documents/figures/historical_airseaflux_correlations_III'+str(i)+'.pdf')

for i,dummy in enumerate(lon):
        cube2 = cubes[i]
        cube2.data[np.where(cube2.data >= 1)] = 1
        cube2.data[np.where(cube2.data <= -1)] = -1
        cube_region_tmp = cube2.intersection(longitude=(lon_west, lon_east))
        cube_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))
        plt.close('all')
        ax = plt.axes(projection=ccrs.PlateCarree())
        #ax.set_global()
        CS = iplt.contourf(cube_region,np.linspace(-1,1,31))
        ax.add_feature(cartopy.feature.LAND,facecolor='#f6f6f6')
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        plt.scatter(lon[i],lat[i],s=300,c = 'k', marker = '*')
        #cbar = plt.colorbar(CS)
        #plt.show(block = False)
        plt.savefig('/home/ph290/Documents/figures/historical_airseaflux_correlations_III'+str(i)+'.png')



'''
cubes = [out_cube1, out_cube2, out_cube3, out_cube4, out_cube6]


for i,dummy in enumerate(lon):
        cube2 = cubes[i]
        cube2.data[np.where(cube2.data >= 1)] = 1
        cube2.data[np.where(cube2.data <= 0)] = 0
	cubes[i].data = cube2.data
	



cube_combined = cubes[0] + cubes[1] +cubes[2] +cubes[3] +cubes[4]

cube_combined.data[np.where(cube_combined.data >= 1)] = 1
cube_combined.data[np.where(cube_combined.data <= -1)] = -1


cube_region_tmp = cube_combined.intersection(longitude=(lon_west, lon_east))
cube_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))

plt.close('all')
ax = plt.axes(projection=ccrs.PlateCarree())
        #ax.set_global()
CS = iplt.contourf(cube_region,np.linspace(-1,1,31))
ax.add_feature(cartopy.feature.LAND,facecolor='#f6f6f6')
ax.coastlines(resolution='50m', color='black', linewidth=1)
for i,dummy in enumerate(lon):
	plt.scatter(lon[i],lat[i],s=100,c = 'k', marker = '*')

cbar = plt.colorbar(CS)
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/airseaflux_correlations_III.png')

'''
