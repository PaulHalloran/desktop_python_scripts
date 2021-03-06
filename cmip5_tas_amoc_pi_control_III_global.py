'''

import iris.plot as iplt
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris
import glob
import iris.experimental.concatenate
import iris.analysis
import iris.quickplot as qplt
import iris.analysis.cartography
import cartopy.crs as ccrs
import subprocess
from iris.coords import DimCoord
import iris.coord_categorisation
import matplotlib as mpl
import gc
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
import monthly_to_yearly as m2yr
from matplotlib import mlab
import matplotlib.mlab as ml
import cartopy
import running_mean
import matplotlib.cm as mpl_cm
import cartopy.feature as cfeature


def digital(cube):
    cube_out = cube.copy()
    cube_out[np.where(cube >= 0.0)] = 1.0
    cube_out[np.where(cube < 0.0)] = -1.0
    return cube_out

def index_of_array_items_in_another(x,y):
	index = np.argsort(x)
	sorted_x = x[index]
	sorted_index = np.searchsorted(sorted_x, y)
	yindex = np.take(index, sorted_index)
	return np.ma.array(yindex)
	
	
variables = np.array(['tas'])
z = np.array([0.05,2.5e-7,0.5])
no_agree = np.array([4,5,2])
# 4 or 6
# 5 of 7
# 2 of 3
#note - not enough good models with stream function and rsds to do that that way - will have to do from ssts

for k,variable in enumerate(variables):

	print '********************************************'
	print '      '+variable
	print '********************************************'

	files1 = np.array(glob.glob('/media/usb_external1/cmip5/msftmyz/piControl/*.nc'))
	files2 = np.array(glob.glob('/media/usb_external1/cmip5/'+variable+'_regridded/*'+variable+'*piControl*.nc'))

	
	#which models do we have?
	models1 = []
	for file in files1:
		models1.append(file.split('/')[-1].split('_')[0])


	models_unique1 = np.unique(np.array(models1))

	models2 = []
	for file in files2:
		models2.append(file.split('/')[-1].split('_')[0])


	models_unique2 = np.unique(np.array(models2))

	models_unique = np.intersect1d(models_unique1,models_unique2)


	
	#read in AMOC
	


	cmip5_max_strmfun = []
	cmip5_year = []

	models_unique = models_unique.tolist()

	# model and no. years
	# 'CCSM4'1011
	# 'CESM1-BGC'460
	# 'CESM1-CAM5'279
	# 'CESM1-FASTCHEM'182
	# 'CNRM-CM5'810
	# 'CNRM-CM5-2'370
	# 'CanESM2'966
	# 'MPI-ESM-LR'960
	# 'MPI-ESM-MR'960
	# 'MPI-ESM-P'1116
	# 'MRI-CGCM3'460


	if variable == 'tas':
		try:
			models_unique.remove('FGOALS-g2')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('inmcm4')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CESM1-WACCM')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CESM1-CAM5')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CESM1-FASTCHEM')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CNRM-CM5-2')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('MPI-ESM-LR')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('MPI-ESM-P')
		except:
			print 'nowt to remove'

	if variable == 'pr':
		try:
			models_unique.remove('FGOALS-g2')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('inmcm4')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CESM1-WACCM')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CESM1-CAM5')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CESM1-FASTCHEM')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CNRM-CM5-2')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('MPI-ESM-LR')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('MPI-ESM-P')
		except:
			print 'nowt to remove'

	if variable == 'rsds':
		try:
			models_unique.remove('FGOALS-g2')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CESM1-WACCM')
		except:
			print 'nowt to remove'
		try:
			models_unique.remove('CESM1-FASTCHEM')
		except:
			print 'nowt to remove'


	for model in models_unique:
		print model
		files = np.array(glob.glob('/media/usb_external1/cmip5/msftmyz/piControl/*'+model+'_*.nc'))
		cube = iris.load_cube(files)[:,0,:,:]
		loc = np.where(cube.coord('grid_latitude').points >= 26.0)[0]
		lat = cube.coord('grid_latitude').points[loc[0]]
		sub_cube = cube.extract(iris.Constraint(grid_latitude = lat))
		stream_function_tmp = sub_cube.collapsed('depth',iris.analysis.MAX)
		coord = stream_function_tmp.coord('time')
		dt = coord.units.num2date(coord.points)
		year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
		tmp = running_mean.running_mean(signal.detrend(stream_function_tmp.data/1.0e9),40)
		cmip5_max_strmfun.append(tmp[np.logical_not(np.isnan(tmp))])
		cmip5_year.append(year_tmp[np.logical_not(np.isnan(tmp))])

	cmip5_max_strmfun = np.array(cmip5_max_strmfun)
	cmip5_year = np.array(cmip5_year)


	#read in variable

	cmip5_max_variable_high = []
	cmip5_max_variable_low = []
	cmip5_max_variable_diff = []
	digital_high_low = np.empty([np.size(models_unique),180,360])

	for i,model in enumerate(models_unique):
		print model
		files = np.array(glob.glob('/media/usb_external1/cmip5/'+variable+'_regridded/*'+model+'_*'+variable+'*piControl*.nc'))
		cube = iris.load_cube(files)
		cube.data = signal.detrend(cube.data,axis = 0)
		coord = cube.coord('time')
		dt = coord.units.num2date(coord.points)
		year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
		high = np.where(cmip5_max_strmfun[i] > np.mean(cmip5_max_strmfun[i])+np.std(cmip5_max_strmfun[i]))
		low = np.where(cmip5_max_strmfun[i] < np.mean(cmip5_max_strmfun[i])-np.std(cmip5_max_strmfun[i]))
		high_yrs = np.intersect1d(year_tmp,cmip5_year[i][high[0]])
		low_yrs = np.intersect1d(year_tmp,cmip5_year[i][low[0]])
		high_yrs_index = np.nonzero(np.in1d(high_yrs,year_tmp))[0]
		low_yrs_index = np.nonzero(np.in1d(low_yrs,year_tmp))[0]
		cmip5_max_variable_high_tmp = cube[high_yrs_index].collapsed('time',iris.analysis.MEAN)
		cmip5_max_variable_low_tmp = cube[low_yrs_index].collapsed('time',iris.analysis.MEAN)
		cmip5_max_variable_high.append(cmip5_max_variable_high_tmp)
		cmip5_max_variable_low.append(cmip5_max_variable_low_tmp)
		cmip5_max_variable_diff.append(cmip5_max_variable_high_tmp - cmip5_max_variable_low_tmp)
		digital_high_low[i,:,:] = digital(cmip5_max_variable_diff[i].data)


	cmip5_max_variable_high = np.array(cmip5_max_variable_high)
	cmip5_max_variable_low = np.array(cmip5_max_variable_low)
	cmip5_max_variable_diff = np.array(cmip5_max_variable_diff)

	mean_pattern = np.mean(cmip5_max_variable_diff,axis = 0)



	digital_high_low_mean_cube = cmip5_max_variable_diff[0].copy()
	digital_high_low_mean_cube.data = digital(mean_pattern.data)

	digital_data_tmp1 = digital_high_low.copy()
	digital_data_tmp2 = digital_high_low.copy()
	digital_data_tmp1[np.where(digital_high_low == 1.0)] = 0
	digital_low = np.sum(digital_data_tmp1,axis = 0)
	digital_data_tmp2[np.where(digital_high_low == -1.0)] = 0
	digital_high = np.sum(digital_data_tmp2,axis = 0)
	digital_low_cube = cmip5_max_variable_diff[0].copy()
	digital_low_cube.data = digital_low
	digital_high_cube = cmip5_max_variable_diff[0].copy()
	digital_high_cube.data = digital_high
	digital_low_cube.data[np.where(digital_high_low_mean_cube.data == 1.0)] = 0
	digital_high_cube.data[np.where(digital_high_low_mean_cube.data == -1.0)] = 0

	digital_cube_final = digital_high_low_mean_cube.copy()
	tmp = np.zeros([digital_high_low_mean_cube.shape[0],digital_high_low_mean_cube.shape[1]])
	tmp[:] = np.nan
	digital_cube_final.data = tmp

	digital_cube_final.data[np.where(digital_low_cube.data <= -1*no_agree[k])] = 1
	digital_cube_final.data[np.where(digital_high_cube.data >= no_agree[k])] = 1



	#dummy cube
	latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
						units='degrees')
	longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
						 units='degrees')
	dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
				dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

	digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')

	north = 90
	south = -90
	east = 180
	west = -180


	tmp1 = mean_pattern.intersection(longitude=(west, east))
	tmp1 = tmp1.intersection(latitude=(south, north))


	tmp2 = digital_cube_final.intersection(longitude=(west, east))
	tmp2 = tmp2.intersection(latitude=(south, north))


	cmap = mpl_cm.get_cmap('coolwarm')

'''

plt.close('all')
fig = plt.figure()
min_val = -1*z[k]
max_val = z[k]
tmp = tmp1.copy()
tmp.data[np.where(tmp.data < min_val)] = min_val*0.9999
tmp.data[np.where(tmp.data > max_val)] = max_val*0.9999
ax1 = fig.subplot(111,projection=ccrs.Miller(central_longitude=-20))
ax1.set_extent([-100, 40, -85, 85], crs=ccrs.PlateCarree())
cb = iplt.contourf(tmp,np.linspace(min_val,max_val,51),cmap=cmap)
points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
plt.gca().coastlines()
ax1.add_feature(cfeature.LAND,facecolor='#f6f6f6')
#ax1.set_global()
#plt.title(low-high nat. amo')
cbar = fig.colorbar(cb, orientation='vertical')
plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/'+variable+'_internal_variability_james.ps')


