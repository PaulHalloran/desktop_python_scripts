'''

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
import running_mean


def index_of_array_items_in_another(x,y):
	index = np.argsort(x)
	sorted_x = x[index]
	sorted_index = np.searchsorted(sorted_x, y)
	yindex = np.take(index, sorted_index, mode="clip")
	mask = x[yindex] != y
	return np.ma.array(yindex, mask=mask)

def digital(cube):
    cube_out = cube.copy()
    cube_out[np.where(cube >= 0.0)] = 1.0
    cube_out[np.where(cube < 0.0)] = -1.0
    return cube_out

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

'''

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

plt.close('all')
plt.figure()
min_val = -2.0
max_val = +2.0
tmp = tmp1.copy()
tmp.data[np.where(tmp.data < min_val)] = min_val
tmp.data[np.where(tmp.data > max_val)] = max_val
qplt.contourf(tmp,np.linspace(min_val,max_val,51))
points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
plt.gca().coastlines()
plt.title('Shortwave: low-high nat. amo >=4of6')
plt.savefig('/home/ph290/Documents/figures/sw_amo_nat.pdf')
plt.show()







