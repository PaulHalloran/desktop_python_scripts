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

directory = '/home/ph290/data0/cmip5_data/regridded'

files = glob.glob(directory+'/fgco2_*_regridded.nc')

models = []
for file in files:
    models.append(file.split('/')[-1].split('_')[1])

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
	cube = iris.load_cube(directory+'/fgco2_'+model+'_regridded.nc')
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
	
	lon_west = -52
	lon_east = -52
	lat_south = 45
	lat_north = 46

	cube_region = cube.intersection(longitude=(lon_west, lon_east))
	cube_newfoundland = cube_region.intersection(latitude=(lat_south, lat_north))
	cube_newfoundland_mean = cube_newfoundland.collapsed(['latitude','longitude'],iris.analysis.MEAN)

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
	b4 = cube_newfoundland_mean.data
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


out_data1_mean = np.mean(out_data1,axis = 0)
out_cube1 = cube[0].copy()
out_cube1.data = out_data1_mean

out_data2_mean = np.mean(out_data2,axis = 0)
out_cube2 = cube[0].copy()
out_cube2.data = out_data2_mean

out_data3_mean = np.mean(out_data3,axis = 0)
out_cube3 = cube[0].copy()
out_cube3.data = out_data3_mean

out_data4_mean = np.mean(out_data4,axis = 0)
out_cube4 = cube[0].copy()
out_cube4.data = out_data4_mean

out_data5_mean = np.mean(out_data5,axis = 0)
out_cube5 = cube[0].copy()
out_cube5.data = out_data5_mean

out_data6_mean = np.mean(out_data6,axis = 0)
out_cube6 = cube[0].copy()
out_cube6.data = out_data6_mean

# qplt.contourf(out_cube1,31)
# plt.show()
# 
# qplt.contourf(out_cube2,31)
# plt.show()
# 
# qplt.contourf(out_cube3,31)
# plt.show()

cube_combined = out_cube1 + out_cube2 + out_cube3 + out_cube4 + out_cube5 + out_cube6

plt.close('all')
qplt.contourf(cube_combined,np.linspace(-1,1,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/airseaflux_correlations_10yr_really_gaussian.png')
#plt.show()

