
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
from iris.coords import DimCoord
import iris.plot as iplt
import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy.ma as ma
import running_mean
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

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
	
#this is a simple function that we call later to look at the file names and extarct from them a unique list of models to process
#note that the model name is in the filename when downlaode ddirectly from the CMIP5 archive
def model_names(directory,variable,experiment):
	files = glob.glob(directory+'/*'+variable+'_'+experiment+'*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models

def index_of_array_items_in_another(x,y):
	index = np.argsort(x)
	sorted_x = x[index]
	sorted_index = np.searchsorted(sorted_x, y)
	yindex = np.take(index, sorted_index, mode="clip")
	mask = x[yindex] != y
	return np.ma.array(yindex, mask=mask)
	
plots = []
dots = []
	
'''
	
'''
#natural
'''

	
variables = np.array(['tas','pr'])
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

	files1 = np.array(glob.glob('/media/usb_external1/cmip5/msftmyz/piControl/*piControl*.nc'))
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

	north = 85
	south = -20
	east = 40
	west = -130


	tmp1 = mean_pattern.intersection(longitude=(west, east))
	tmp1 = tmp1.intersection(latitude=(south, north))


	tmp2 = digital_cube_final.intersection(longitude=(west, east))
	tmp2 = tmp2.intersection(latitude=(south, north))


	plots.append(tmp1.copy())
	dots.append(tmp2.copy())

# 	cmap = mpl_cm.get_cmap('coolwarm')
# 
# 	plt.close('all')
# 	plt.figure()
# 	min_val = -1*z[k]
# 	max_val = z[k]
# 	tmp = tmp1.copy()
# 	tmp.data[np.where(tmp.data < min_val)] = min_val*0.9999
# 	tmp.data[np.where(tmp.data > max_val)] = max_val*0.9999
# 	qplt.contourf(tmp,np.linspace(min_val,max_val,51),cmap=cmap)
# 	points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
# 	plt.gca().coastlines()
# 	plt.title(variable+': low-high nat. amo')
#  	plt.show()
#	plt.savefig('/home/ph290/Documents/figures/'+variable+'_internal_variability.ps')

'''
#natural from ssts fro SW because not enough models to do AMOC
'''


#first process: /data/temp/ph290/last_1000

input_directory = '/media/usb_external1/cmip5/sw_regridded/'
input_directory2 = '/media/usb_external1/cmip5/reynolds_data/'

variables = np.array(['rsds','tos'])
experiments = ['piControl','past1000','sstClim','sstClimAerosol']

'''
#Main bit of code follows...
'''

'''
#SSTs to calculate models' AMO
'''

experiment = experiments[0]
variable = variables[1]

models = model_names(input_directory2,variable,experiment)

models = list(models)
#models.remove('bcc-csm1-1')


lon_west1 = -75.0+360
lon_east1 = -7.5+360
lat_south1 = 0.0
lat_north1 = 60.0
region1 = iris.Constraint(longitude=lambda v: lon_west1 <= v <= lon_east1,latitude=lambda v: lat_south1 <= v <= lat_north1)

cube1 = iris.load_cube(input_directory2+models[0]+'*'+variable+'*.nc')[0]

models2_amo = []
#cubes = []
ts_amo = []
yrs_amo = []

for model in models:
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory2+model+'*'+variable+'_'+experiment+'*.nc')
	except:
		cube = iris.load(input_directory2+model+'*'+variable+'_'+experiment+'*.nc')
		cube = cube[0]
	#cubes.append(cube)
	tmp1 = cube.extract(region1)
	try:
		tmp1.coord('latitude').guess_bounds()
	except:
		print 'has bounds'
	try:
		tmp1.coord('longitude').guess_bounds()
	except:
		print 'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(tmp1)
	tmp2 = tmp1.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas)
	#print np.shape(tmp2)
	ts_amo.append(tmp2.data)
	coord = tmp2.coord('time')
	dt = coord.units.num2date(coord.points)
	tmp_yrs = np.array([coord.units.num2date(value).year for value in coord.points])
	yrs_amo.append(tmp_yrs)
	models2_amo.append(model)
	
	
'''
#models' sw
'''

experiment = experiments[0]
variable = variables[0]

models = model_names(input_directory,variable,experiment)

models = list(models)
#models.remove('bcc-csm1-1')

cube1 = iris.load_cube(input_directory+models[0]+'*'+variable+'*.nc')[0]

models2_sw= []
cubes = []
yrs_sw = []

for model in models:
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory+model+'*'+variable+'_'+experiment+'*.nc')
	except:
		cube = iris.load(input_directory+model+'*'+variable+'_'+experiment+'*.nc')
		cube = cube[0]
	cubes.append(cube)
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	tmp_yrs = np.array([coord.units.num2date(value).year for value in coord.points])
	yrs_sw.append(tmp_yrs)
	models2_sw.append(model)




'''
#main processing
'''

common_models = np.intersect1d(models2_amo, models2_sw)

sw_high_amo = []
sw_low_amo = []
sw_low_minus_high_amo = []
high_low = np.empty([np.size(common_models),np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
digital_high_low = high_low.copy()

for i,model in enumerate(common_models):
	loc1 = np.where(np.array(models2_amo) == np.array(model))[0][0]
	loc2 = np.where(np.array(models2_sw) == np.array(model))[0][0]
	amo_tmp = signal.detrend(ts_amo[loc1])
	high_amo_yrs = yrs_amo[loc1][np.where(amo_tmp > np.mean(amo_tmp))]
	low_amo_yrs = yrs_amo[loc1][np.where(amo_tmp < np.mean(amo_tmp))]
	common_high_yrs = np.intersect1d(yrs_sw[loc2], high_amo_yrs)
	common_low_yrs = np.intersect1d(yrs_sw[loc2], low_amo_yrs)
	common_high_yrs_index = index_of_array_items_in_another(yrs_sw[loc2],common_high_yrs)
	common_low_yrs_index = index_of_array_items_in_another(yrs_sw[loc2],common_low_yrs)
	sw_high_amo.append(cubes[loc2][common_high_yrs_index].collapsed('time',iris.analysis.MEAN))
	sw_low_amo.append(cubes[loc2][common_low_yrs_index].collapsed('time',iris.analysis.MEAN))
	sw_low_minus_high_amo = sw_low_amo[i] - sw_high_amo[i]
	high_low[i,:,:] = sw_low_minus_high_amo.data
	digital_high_low[i,:,:] = digital(sw_low_minus_high_amo.data)



high_low_mean_cube = cubes[0][0].copy()
digital_high_low_mean_cube = cubes[0][0].copy()
high_low_mean_cube.data = np.mean(high_low,axis = 0)
digital_high_low_mean_cube.data = digital(np.mean(high_low,axis = 0))

#processing digital fields. Count number of models which are +ve where the mean is +ve and -ve where the mean is -ve, then turn that into stippling
digital_data_tmp1 = digital_high_low.copy()
digital_data_tmp2 = digital_high_low.copy()
digital_data_tmp1[np.where(digital_high_low == 1.0)] = 0
digital_low = np.sum(digital_data_tmp1,axis = 0)
digital_data_tmp2[np.where(digital_high_low == -1.0)] = 0
digital_high = np.sum(digital_data_tmp2,axis = 0)
digital_low_cube = cubes[0][0].copy()
digital_low_cube.data = digital_low
digital_high_cube = cubes[0][0].copy()
digital_high_cube.data = digital_high
digital_low_cube.data[np.where(digital_high_low_mean_cube.data == 1.0)] = 0
digital_high_cube.data[np.where(digital_high_low_mean_cube.data == -1.0)] = 0

digital_cube_final = digital_high_low_mean_cube.copy()
tmp = np.zeros([digital_high_low_mean_cube.shape[0],digital_high_low_mean_cube.shape[1]])
tmp[:] = np.nan
digital_cube_final.data = tmp



digital_cube_final.data[np.where(digital_low_cube.data <= -5)] = 1
digital_cube_final.data[np.where(digital_high_cube.data >= 5)] = 1


#dummy cube
latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
                    units='degrees')
longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
                     units='degrees')
dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')

north = 85
south = -20
east = 40
west = -130


tmp1 = high_low_mean_cube.intersection(longitude=(west, east))
tmp1 = tmp1.intersection(latitude=(south, north))

tmp2 = digital_cube_final.intersection(longitude=(west, east))
tmp2 = tmp2.intersection(latitude=(south, north))

plots.append(tmp1.copy())
dots.append(tmp2.copy())


'''

'''
#aerosol
'''


input_directory = '/media/usb_external1/cmip5/aero/'

variables = np.array(['rsds','tas','pr'])
variable_names = np.array(['shortwave','surface_temp','precipitation'])
z_range_maxs = [4.8,1.0,0.0000025]
experiments = ['historicalMisc']

'''
#Main bit of code follows...
'''

	
'''
#models' sw 
'''

'''

debug_cube = []

for k,variable in enumerate(variables):

	#k=0
	models = model_names(input_directory,variable,experiments[0])

	models = list(models)
	#models.remove('CSIRO-Mk3-6-0')



	cube1 = iris.load_cube(input_directory+models[0]+'*'+variable+'*'+experiments[0]+'_*.nc')

	models2_sw= []
	cubes1 = []
	cubes2 = []
	cubes_diff = []
	high_low = np.empty([np.size(models),np.shape(cube1[0].data)[0],np.shape(cube1[0].data)[1]])
	digital_high_low = high_low.copy()

	for i,model in enumerate(models):
		print 'processing: '+model
		try:
			cube = iris.load_cube(input_directory+model+'*'+variable+'_'+experiments[0]+'_*.nc')
		except:
			cube = iris.load(input_directory+model+'*'+variable+'_'+experiments[0]+'_*.nc')
			cube = cube[0]
		if variable == 'rsds':
			debug_cube.append(cube)
		cube.data = signal.detrend(cube.data,axis=0)
		coord = cube.coord('time')
		dt = coord.units.num2date(coord.points)
		year = np.array([coord.units.num2date(value).year for value in coord.points])
		high_amo_loc = np.where((year >= 1930) & (year < 1960))[0]
		low_amo_loc = np.where((year >= 1970) & (year < 1990))[0]
		cube_tmp_h = cube[high_amo_loc].collapsed('time',iris.analysis.MEAN)
		cube_tmp_l = cube[low_amo_loc].collapsed('time',iris.analysis.MEAN)
		cubes1.append(cube_tmp_h)
		cubes2.append(cube_tmp_l)
		models2_sw.append(model)
		cubes_diff.append(cubes2[i]-cubes1[i])
# 		fig = plt.subplots(3,1)
# 		ax1 = plt.subplot(3,1,1)
# 		qplt.contourf(cube_tmp_h,31)
# 		plt.gca().coastlines()
# 		plt.title(model + '1930-1960 sw')
# 		ax2 = plt.subplot(3,1,2)
# 		qplt.contourf(cube_tmp_h,31)
# 		plt.gca().coastlines()
# 		plt.title('1970-1990')
# 		ax3 = plt.subplot(3,1,3)
# 		qplt.contourf(cubes_diff[i],31)
# 		plt.gca().coastlines()
# 		plt.title('1970-1990 minus 1930-1960')
# 		#plt.show()
# 		plt.savefig('/home/ph290/Documents/figures/sw_test_'+model+'_detrended.png',dpi = 600)
	
		high_low[i,:,:] = cubes_diff[i].data
		digital_high_low[i,:,:] = digital(cubes_diff[i].data)


	high_low_mean_cube = cube1[0].copy()
	digital_high_low_mean_cube = cube1[0].copy()
	high_low_mean_cube.data = np.mean(high_low,axis = 0)
	digital_high_low_mean_cube.data = digital(np.mean(high_low,axis = 0))


	#processing digital fields. Count number of models which are +ve where the mean is +ve and -ve where the mean is -ve, then turn that into stippling
	digital_data_tmp1 = digital_high_low.copy()
	digital_data_tmp2 = digital_high_low.copy()
	digital_data_tmp1[np.where(high_low == 1.0)] = 0
	digital_low = np.sum(digital_data_tmp1,axis = 0)
	digital_data_tmp2[np.where(high_low == -1.0)] = 0
	digital_high = np.sum(digital_data_tmp2,axis = 0)
	digital_low_cube = cube1[0].copy()
	digital_low_cube.data = digital_low
	digital_high_cube = cube1[0].copy()
	digital_high_cube.data = digital_high
	digital_low_cube.data[np.where(digital_high_low_mean_cube.data == 1.0)] = 0
	digital_high_cube.data[np.where(digital_high_low_mean_cube.data == -1.0)] = 0

	digital_cube_final = digital_high_low_mean_cube.copy()
	tmp = np.zeros([digital_high_low_mean_cube.shape[0],digital_high_low_mean_cube.shape[1]])
	tmp[:] = np.nan
	digital_cube_final.data = tmp


	#!!!!!!!!!!!#
	no_agree = 2
	#!!!!!!!!!!!#
	digital_cube_final.data[np.where(digital_low_cube.data <= -1.0*no_agree)] = 1
	digital_cube_final.data[np.where(digital_high_cube.data >= no_agree)] = 1



	#dummy cube
	latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
				units='degrees')
	longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
				 units='degrees')
	dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
			dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

	digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')

	north = 85
	south = -20
	east = 40
	west = -130

	tmp1 = high_low_mean_cube.intersection(longitude=(west, east))
	tmp1 = tmp1.intersection(latitude=(south, north))

	tmp2 = digital_cube_final.intersection(longitude=(west, east))
	tmp2 = tmp2.intersection(latitude=(south, north))

	plots.append(tmp1.copy())
	dots.append(tmp2.copy())

	# 	plt.close('all')
	# 	plt.figure()
	# 	min_val = -1.0*z_range_maxs[k]
	# 	max_val = z_range_maxs[k]
	# 	tmp = tmp1.copy()
	# 	tmp.data[np.where(tmp.data < min_val)] = min_val
	# 	tmp.data[np.where(tmp.data > max_val)] = max_val
	# 	qplt.contourf(tmp,np.linspace(min_val,max_val,51))
	# 	points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
	# 	plt.gca().coastlines()
	# 	plt.title(variable_names[k]+' aero 1960-1980 minus 1935-1955 '+np.str(no_agree)+'of3')
	# # 	plt.savefig('/home/ph290/Documents/figures/'+variable_names[k]+'_aero.pdf')
	# 	plt.show()


###################
#debugging
###################

cube = debug_cube[0]
try:
	cube.coord('latitude').guess_bounds()
	cube.coord('longitude').guess_bounds()
except:
	print ' '
grid_areas = iris.analysis.cartography.area_weights(cube)
a = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
coord = a.coord('time')
dt = coord.units.num2date(coord.points)
a_year = np.array([coord.units.num2date(value).year for value in coord.points])

cube = debug_cube[1] 
try:
	cube.coord('latitude').guess_bounds()
	cube.coord('longitude').guess_bounds()
except:
	print ' '
grid_areas = iris.analysis.cartography.area_weights(cube)
b = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
coord = b.coord('time')
dt = coord.units.num2date(coord.points)
b_year = np.array([coord.units.num2date(value).year for value in coord.points])

cube = debug_cube[2] 
try:
	cube.coord('latitude').guess_bounds()
	cube.coord('longitude').guess_bounds()
except:
	print ' '
grid_areas = iris.analysis.cartography.area_weights(cube)
c = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
coord = c.coord('time')
dt = coord.units.num2date(coord.points)
c_year = np.array([coord.units.num2date(value).year for value in coord.points])

plt.close('all')
plt.figure
plt.plot(a_year,a.data)
plt.plot(b_year,b.data)
plt.plot(c_year,c.data)
plt.savefig('/home/ph290/Documents/figures/test/test.png')
plt.show()

'''

x=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']
for j,i in enumerate(np.round(np.linspace(0,160,8))):
	print i
	plt.figure()
	qplt.contourf(debug_cube[2][i:i+10].collapsed('time',iris.analysis.MEAN),np.linspace(-8,8,17))
	plt.gca().coastlines()
	plt.title(np.str(1860+i)+' to '+np.str(1860+i+20))
	plt.savefig('/home/ph290/Documents/figures/test/plt_'+x[j]+'.png',dpi=60)
	plt.close('all')


'''

'''
#volcanic
'''

input_directory = '/media/usb_external1/cmip5/tas_regridded/' # tas

variables = np.array(['tas'])
experiments = ['past1000']


#tas:

experiment = experiments[0]

models = model_names(input_directory,variables[0],experiment)

models = list(models)
#models.remove('bcc-csm1-1')

cube1 = iris.load_cube(input_directory+models[0]+'*'+variables[0]+'*'+experiment+'*.nc')[0]

models2 = []
cubes = []

for model in models:
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory+model+'*'+variables[0]+'_'+experiment+'*.nc')
	except:
		cube = iris.load(input_directory+model+'*'+variables[0]+'_'+experiment+'*.nc')
		cube = cube[0]
	cubes.append(cube)


coord = cubes[0].coord('time')
dt = coord.units.num2date(coord.points)
model_years = np.array([coord.units.num2date(value).year for value in coord.points])

'''
#Volcanic forcing
'''

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)

data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]

data = np.mean(data_tmp,axis = 1)
data_final = data1.copy()
data_final[:,1] = data

volcanic_smoothing = 5 #yrs

volc_years = data_final[:,0]

loc = np.where((volc_years >= 850) & (volc_years <= 1849))[0]

volc_tmp = running_mean_post.running_mean_post(data_final[loc,1],36.0*volcanic_smoothing)


volc_years = data_final[np.arange(np.size(volc_tmp)),0]
yrs = np.floor(data_final[np.arange(np.size(volc_tmp)),0])
yrs_unique = np.unique(yrs)
data_ann = np.empty([np.size(yrs_unique),2])

for i,y in enumerate(yrs_unique):
        loc = np.where(yrs == y)[0]
        data_ann[i,0] = y
        data_ann[i,1] = np.mean(volc_tmp[loc])

volc = data_ann[:,1]

volc_mean = np.mean(volc)
high_volc = np.where(volc > volc_mean)[0]
low_volc = np.where(volc < volc_mean)[0]

high_volc_data = np.empty([np.size(cubes),np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
low_volc_data = high_volc_data.copy()
change_volc_data = high_volc_data.copy()
digital_volc_data = high_volc_data.copy()
digital_high = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
digital_low = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])

for i,cube in enumerate(cubes):
	tmp = cube[high_volc].collapsed('time',iris.analysis.MEAN)
	high_volc_data[i,:,:] = tmp.data
	tmp2 = cube[low_volc].collapsed('time',iris.analysis.MEAN)
	low_volc_data[i,:,:] = tmp2.data
	change_volc_data[i,:,:] = tmp.data - tmp2.data
	digital_volc_data[i,:,:] = digital(change_volc_data[i,:,:])

digital_volc_data_tmp1 = digital_volc_data.copy()
digital_volc_data_tmp2 = digital_volc_data.copy()
digital_volc_data_tmp1[np.where(digital_volc_data == 1.0)] = 0
digital_low = np.sum(digital_volc_data_tmp1,axis = 0)
digital_volc_data_tmp2[np.where(digital_volc_data == -1.0)] = 0
digital_high = np.sum(digital_volc_data_tmp2,axis = 0)

high_volc_data_mean = np.mean(high_volc_data,axis = 0)
low_volc_data_mean = np.mean(low_volc_data,axis = 0)
change_volc_data_mean = np.mean(change_volc_data,axis = 0)

high_volc_data_mean_cube = cubes[0][0].copy()
high_volc_data_mean_cube.data = high_volc_data_mean
low_volc_data_mean_cube = cubes[0][0].copy()
low_volc_data_mean_cube.data = low_volc_data_mean
change_volc_data_mean_cube = cubes[0][0].copy()
change_volc_data_mean_cube.data = change_volc_data_mean
digital_low_cube = cubes[0][0].copy()
digital_low_cube.data = digital_low
digital_high_cube = cubes[0][0].copy()
digital_high_cube.data = digital_high

change_digital = change_volc_data_mean_cube.copy()
change_digital.data = digital(change_volc_data_mean_cube.data)

digital_low_cube.data[np.where(change_digital.data == 1.0)] = 0
digital_high_cube.data[np.where(change_digital.data == -1.0)] = 0

digital_cube_final = digital_low_cube.copy()
tmp = np.zeros([digital_cube_final.shape[0],digital_cube_final.shape[1]])
tmp[:] = np.nan
digital_cube_final.data = tmp

#!!!!!!!!!!!#
no_agree = 6
#!!!!!!!!!!!#

digital_cube_final.data[np.where(digital_low_cube.data <= -1.0*no_agree)] = 1
digital_cube_final.data[np.where(digital_high_cube.data >= no_agree)] = 1


#dummy cube
latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
                    units='degrees')
longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
                     units='degrees')
dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')

north = 85
south = -20
east = 40
west = -130

tmp1 = change_volc_data_mean_cube.intersection(longitude=(west, east))
tmp1 = tmp1.intersection(latitude=(south, north))

tmp2 = digital_cube_final.intersection(longitude=(west, east))
tmp2 = tmp2.intersection(latitude=(south, north))

plots.append(tmp1.copy())
dots.append(tmp2.copy())

# plt.close('all')
# plt.figure()
# minz = -0.3
# maxz = 0.3
# tmpb = tmp.copy()
# tmpb.data[np.where(tmpb.data < minz)] = minz
# tmpb.data[np.where(tmpb.data > maxz)] = maxz
# qplt.contourf(tmpb,np.linspace(minz,maxz,51))
# points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
# plt.gca().coastlines()
# plt.title('tamperature: high-low volcanism >= '+np.str(no_agree)+' of '+np.str(np.size(cubes)))
# #plt.show(block = False)
# plt.savefig('/home/ph290/Documents/figures/tas_volcanoes.pdf')

#pr:

input_directory = '/media/usb_external1/cmip5/reynolds_data/past1000/' # tas

variables = np.array(['pr'])
experiments = ['past1000']
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.

'''
#Main bit of code follows...
'''

experiment = experiments[0]

models = model_names(input_directory,variables[0],experiment)

models = list(models)
#models.remove('bcc-csm1-1')

cube1 = iris.load_cube(input_directory+models[0]+'*'+variables[0]+'*.nc')[0]

models2 = []
cubes = []

for model in models:
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory+model+'*'+variables[0]+'_'+experiment+'*.nc')
	except:
		cube = iris.load(input_directory+model+'*'+variables[0]+'_'+experiment+'*.nc')
		cube = cube[0]
	cubes.append(cube)


coord = cubes[0].coord('time')
dt = coord.units.num2date(coord.points)
model_years = np.array([coord.units.num2date(value).year for value in coord.points])

'''
#Volcanic forcing
'''

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)

data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]

data = np.mean(data_tmp,axis = 1)
data_final = data1.copy()
data_final[:,1] = data

volcanic_smoothing = 5 #yrs

volc_years = data_final[:,0]

loc = np.where((volc_years >= 850) & (volc_years <= 1849))[0]

volc_tmp = running_mean_post.running_mean_post(data_final[loc,1],36.0*volcanic_smoothing)


volc_years = data_final[np.arange(np.size(volc_tmp)),0]
yrs = np.floor(data_final[np.arange(np.size(volc_tmp)),0])
yrs_unique = np.unique(yrs)
data_ann = np.empty([np.size(yrs_unique),2])

for i,y in enumerate(yrs_unique):
        loc = np.where(yrs == y)[0]
        data_ann[i,0] = y
        data_ann[i,1] = np.mean(volc_tmp[loc])

volc = data_ann[:,1]

volc_mean = np.mean(volc)
high_volc = np.where(volc > volc_mean)[0]
low_volc = np.where(volc < volc_mean)[0]

high_volc_data = np.empty([np.size(cubes),np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
low_volc_data = high_volc_data.copy()
change_volc_data = high_volc_data.copy()
digital_volc_data = high_volc_data.copy()
digital_high = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
digital_low = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])

for i,cube in enumerate(cubes):
	tmp = cube[high_volc].collapsed('time',iris.analysis.MEAN)
	high_volc_data[i,:,:] = tmp.data
	tmp2 = cube[low_volc].collapsed('time',iris.analysis.MEAN)
	low_volc_data[i,:,:] = tmp2.data
	change_volc_data[i,:,:] = tmp.data - tmp2.data
	digital_volc_data[i,:,:] = digital(change_volc_data[i,:,:])

digital_volc_data_tmp1 = digital_volc_data.copy()
digital_volc_data_tmp2 = digital_volc_data.copy()
digital_volc_data_tmp1[np.where(digital_volc_data == 1.0)] = 0
digital_low = np.sum(digital_volc_data_tmp1,axis = 0)
digital_volc_data_tmp2[np.where(digital_volc_data == -1.0)] = 0
digital_high = np.sum(digital_volc_data_tmp2,axis = 0)

high_volc_data_mean = np.mean(high_volc_data,axis = 0)
low_volc_data_mean = np.mean(low_volc_data,axis = 0)
change_volc_data_mean = np.mean(change_volc_data,axis = 0)

high_volc_data_mean_cube = cubes[0][0].copy()
high_volc_data_mean_cube.data = high_volc_data_mean
low_volc_data_mean_cube = cubes[0][0].copy()
low_volc_data_mean_cube.data = low_volc_data_mean
change_volc_data_mean_cube = cubes[0][0].copy()
change_volc_data_mean_cube.data = change_volc_data_mean
digital_low_cube = cubes[0][0].copy()
digital_low_cube.data = digital_low
digital_high_cube = cubes[0][0].copy()
digital_high_cube.data = digital_high

change_digital = change_volc_data_mean_cube.copy()
change_digital.data = digital(change_volc_data_mean_cube.data)

digital_low_cube.data[np.where(change_digital.data == 1.0)] = 0
digital_high_cube.data[np.where(change_digital.data == -1.0)] = 0

digital_cube_final = digital_low_cube.copy()
tmp = np.zeros([digital_cube_final.shape[0],digital_cube_final.shape[1]])
tmp[:] = np.nan
digital_cube_final.data = tmp

#!!!!!!!!!!!#
no_agree = 3
#!!!!!!!!!!!#
digital_cube_final.data[np.where(digital_low_cube.data <= -1.0*no_agree)] = 1
digital_cube_final.data[np.where(digital_high_cube.data >= no_agree)] = 1


#dummy cube
latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
                    units='degrees')
longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
                     units='degrees')
dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')


north = 85
south = -20
east = 40
west = -130

tmp1 = change_volc_data_mean_cube.intersection(longitude=(west, east))
tmp1 = tmp1.intersection(latitude=(south, north))

tmp2 = digital_cube_final.intersection(longitude=(west, east))
tmp2 = tmp2.intersection(latitude=(south, north))

plots.append(tmp1.copy())
dots.append(tmp2.copy())
# 
# 
# plt.close('all')
# plt.figure()
# minz = -0.0000025
# maxz = 0.0000025
# tmpb = tmp.copy()
# tmpb.data[np.where(tmpb.data < minz)] = minz
# tmpb.data[np.where(tmpb.data > maxz)] = maxz
# qplt.contourf(tmpb,np.linspace(minz,maxz,51))
# points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
# plt.gca().coastlines()
# plt.title('precip: high-low volcanism >= '+np.str(no_agree)+' of '+np.str(np.size(cubes)))
# #plt.show(block = False)
# plt.savefig('/home/ph290/Documents/figures/pr_volcanoes.pdf')

#rsds:

input_directory = '/media/usb_external1/cmip5/sw_regridded/'

variables = np.array(['rsds'])
experiments = ['piControl','past1000','sstClim','sstClimAerosol']
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.

'''
#Main bit of code follows...
'''

experiment = experiments[1]

models = model_names(input_directory,variables[0],experiment)

models = list(models)
#models.remove('bcc-csm1-1')
#models.remove('CanESM2')
#models.remove('MIROC5')
#models.remove('MRI-CGCM3')
#models.remove('CSIRO-Mk3-6-0')
import running_mean

cube1 = iris.load_cube(input_directory+models[0]+'*'+variables[0]+'*.nc')[0]

models2 = []
cubes = []

for model in models:
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory+model+'*'+variables[0]+'_'+experiment+'*.nc')
	except:
		cube = iris.load(input_directory+model+'*'+variables[0]+'_'+experiment+'*.nc')
		cube = cube[0]
	cubes.append(cube)


coord = cubes[0].coord('time')
dt = coord.units.num2date(coord.points)
model_years = np.array([coord.units.num2date(value).year for value in coord.points])


'''
#Volcanic forcing
'''

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]
data = np.mean(data_tmp,axis = 1)
data_final = data1.copy()
data_final[:,1] = data

volcanic_smoothing = 5 #yrs

volc_years = data_final[:,0]

loc = np.where((volc_years >= np.min(model_years)) & (volc_years <= np.max(model_years)))[0]

volc_tmp = running_mean_post.running_mean_post(data_final[loc,1],36.0*volcanic_smoothing)


volc_years = data_final[np.arange(np.size(volc_tmp)),0]
yrs = np.floor(data_final[np.arange(np.size(volc_tmp)),0])
yrs_unique = np.unique(yrs)
data_ann = np.empty([np.size(yrs_unique),2])

for i,y in enumerate(yrs_unique):
        loc = np.where(yrs == y)[0]
        data_ann[i,0] = y
        data_ann[i,1] = np.mean(volc_tmp[loc])

volc = data_ann[:,1]

volc_mean = np.mean(volc)
high_volc = np.where(volc > volc_mean)[0]
low_volc = np.where(volc < volc_mean)[0]

high_volc_data = np.empty([np.size(cubes),np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
low_volc_data = high_volc_data.copy()
change_volc_data = high_volc_data.copy()
digital_volc_data = high_volc_data.copy()
digital_high = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
digital_low = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])

for i,cube in enumerate(cubes):
	tmp = cube[high_volc].collapsed('time',iris.analysis.MEAN)
	high_volc_data[i,:,:] = tmp.data
	tmp2 = cube[low_volc].collapsed('time',iris.analysis.MEAN)
	low_volc_data[i,:,:] = tmp2.data
	change_volc_data[i,:,:] = tmp.data - tmp2.data
	digital_volc_data[i,:,:] = digital(change_volc_data[i,:,:])

digital_volc_data_tmp1 = digital_volc_data.copy()
digital_volc_data_tmp2 = digital_volc_data.copy()
digital_volc_data_tmp1[np.where(digital_volc_data == 1.0)] = 0
digital_low = np.sum(digital_volc_data_tmp1,axis = 0)
digital_volc_data_tmp2[np.where(digital_volc_data == -1.0)] = 0
digital_high = np.sum(digital_volc_data_tmp2,axis = 0)

high_volc_data_mean = np.mean(high_volc_data,axis = 0)
low_volc_data_mean = np.mean(low_volc_data,axis = 0)
change_volc_data_mean = np.mean(change_volc_data,axis = 0)

high_volc_data_mean_cube = cubes[0][0].copy()
high_volc_data_mean_cube.data = high_volc_data_mean
low_volc_data_mean_cube = cubes[0][0].copy()
low_volc_data_mean_cube.data = low_volc_data_mean
change_volc_data_mean_cube = cubes[0][0].copy()
change_volc_data_mean_cube.data = change_volc_data_mean
digital_low_cube = cubes[0][0].copy()
digital_low_cube.data = digital_low
digital_high_cube = cubes[0][0].copy()
digital_high_cube.data = digital_high

change_digital = change_volc_data_mean_cube.copy()
change_digital.data = digital(change_volc_data_mean_cube.data)

digital_low_cube.data[np.where(change_digital.data == 1.0)] = 0
digital_high_cube.data[np.where(change_digital.data == -1.0)] = 0

digital_cube_final = digital_low_cube.copy()
tmp = np.zeros([digital_cube_final.shape[0],digital_cube_final.shape[1]])
tmp[:] = np.nan
digital_cube_final.data = tmp

digital_cube_final.data[np.where(digital_low_cube.data <= -6)] = 1
digital_cube_final.data[np.where(digital_high_cube.data >= 6)] = 1


#dummy cube
latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
                    units='degrees')
longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
                     units='degrees')
dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')

north = 85
south = -20
east = 40
west = -130

tmp1 = change_volc_data_mean_cube.intersection(longitude=(west, east))
tmp1 = tmp1.intersection(latitude=(south, north))

tmp2 = digital_cube_final.intersection(longitude=(west, east))
tmp2 = tmp2.intersection(latitude=(south, north))

plots.append(tmp1.copy())
dots.append(tmp2.copy())
# 
# 
# plt.close('all')
# plt.figure()
# min_val = -2.0
# max_val = +2.0
# tmp = tmp1.copy()
# tmp.data[np.where(tmp.data < min_val)] = min_val
# tmp.data[np.where(tmp.data > max_val)] = max_val
# qplt.contourf(tmp,np.linspace(min_val,max_val,51))
# points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
# plt.gca().coastlines()
# plt.title('Shortwave: high-low volcanism >=6of9')
# plt.savefig('/home/ph290/Documents/figures/sw_volcanoes.pdf')
# #plt.show()



'''
#plotting all figures
'''

#vals = np.array([0.05,2.5e-7,0.5,4.8,0.05,2.5e-7,0.05,2.5e-7,1.0])

vals = np.array([0.5,2.5e-6,5.0,0.05,2.5e-7,2.5,0.25,1.0e-6,1.0])


plt.close('all')
cmap = mpl_cm.get_cmap('coolwarm')
index = np.array([4,5,3,0,1,2,6,7,8])

fig = plt.subplots(3,3)
    
for i,dummy in enumerate(plots):
	ax1 = plt.subplot(3,3,i+1)
	min_val = -1.0*vals[i]
	max_val = vals[i]
	tmp = plots[index[i]].copy()
	tmp.data[np.where(tmp.data < min_val)] = min_val*0.999999
	tmp.data[np.where(tmp.data > max_val)] = max_val*0.999999
	CS = iplt.contourf(tmp,np.linspace(min_val,max_val,21),cmap=cmap)
	#plt.title(np.str(i))
	tmp2 = dots[index[i]].copy()
	points = iplt.points(tmp2, c = tmp2.data , s= 0.1)
	plt.gca().coastlines()
	CB = plt.colorbar(CS, shrink=0.8, extend='both',orientation="horizontal")
	plt.subplots_adjust(hspace = .05)
	plt.subplots_adjust(wspace = .025)

# fig.tight_layout()
#plt.show()
plt.savefig('/home/ph290/Documents/figures/palaeo_amo_prop_fig4.pdf')

'''

