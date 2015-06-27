import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import matplotlib.cm as mpl_cm
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


def model_names(directory):
	files = glob.glob(directory+'*uo*rcp85*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models


models = model_names('/media/usb_external1/cmip5/gulf_stream_analysis/regridded/')
remove = np.where(models == 'CCSM4')
#remove inmcm4 because goes funny in the regridding...
models = list(models)
models.pop(remove[0])
models = np.array(models)

runs = 'rcp85'
variables = 'uo'

models_gs_lat = [] #timeseries for each model of teh gulf stream latitude
models_years = [] #the years for each model correstonding to the above

for model in models:
	print model+' calculating gulf stream possition'
	piControl_uo_file = '/media/usb_external1/cmip5/gulf_stream_analysis/regridded/'+model+'_'+variables+'_'+runs+'_regridded.nc'
	piControl_uo_cube = iris.load_cube(piControl_uo_file)
	coord_names = [coord.name() for coord in piControl_uo_cube.coords()]
	test1 = np.size(np.where(np.array(coord_names) == 'ocean sigma coordinate'))
	test1b = np.size(np.where(np.array(coord_names) == 'ocean sigma over z coordinate'))
	if test1 == 1:
		piControl_uo_cube.coord('ocean sigma coordinate').long_name = 'depth'


	if test1b == 1:
		piControl_uo_cube.coord('ocean sigma over z coordinate').long_name = 'depth'


	if np.size(piControl_uo_cube.coords()) >= 4:
		piControl_uo_cube = piControl_uo_cube.extract(iris.Constraint(depth = 0))


#	piControl_uo_cube_50w = piControl_uo_cube.extract(iris.Constraint(longitude = lambda cell: 360-65.0 < cell < 360-50.0))
#	piControl_uo_cube_50w = piControl_uo_cube_50w.extract(iris.Constraint(latitude = lambda cell: 25 < cell < 50))
	piControl_uo_cube_50w = piControl_uo_cube.extract(iris.Constraint(longitude = lambda cell: 360-45.0 < cell < 360-35.0))
	piControl_uo_cube_50w = piControl_uo_cube_50w.extract(iris.Constraint(latitude = lambda cell: 30 < cell < 65))
	#regridding to 0.2 degree grid to make the latitudinal variations in gulf stream smoother
	# latitude = DimCoord(np.arange(20, 65, 0.2), standard_name='latitude',
	# 		    units='degrees')
	# longitude = DimCoord(np.arange(360-44, 360-35, 0.2), standard_name='longitude',
	# 		     units='degrees')
	# cube = iris.cube.Cube(np.zeros(((65-20)*5,((360-35)-(360-44))*5), np.float32),dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
	# cube2 = iris.analysis.interpolate.regrid(piControl_uo_cube_50w,cube, mode='bilinear')
	# piControl_uo_cube_50w = cube2.extract(iris.Constraint(latitude = lambda cell: 30 < cell < 60))
	# piControl_uo_cube_50w.data[np.where(piControl_uo_cube_50w.data >= 1.0e2)] = np.nan
	# qplt.contourf(piControl_uo_cube_50w[0],51)
	# plt.show()
	piControl_uo_cube_50w = piControl_uo_cube_50w.collapsed('longitude',iris.analysis.MEAN)
	latitudes = piControl_uo_cube_50w.coord('latitude').points
	coord = piControl_uo_cube_50w.coord('time')
	years = np.array([coord.units.num2date(value).year for value in coord.points])
	data = piControl_uo_cube_50w.data
	max_current = np.max(data,axis = 1)
	gulf_stream_lat = np.empty(years.size)
	for i in range(years.size):
		gulf_stream_lat[i] = latitudes[np.where(data[i,:] == max_current[i])][0]
	mean_gs_lat = gulf_stream_lat.mean()
	models_years.append(np.array(years))
	models_gs_lat.append(np.array(gulf_stream_lat))

models_years = np.array(models_years)
models_gs_lat = np.array(models_gs_lat)

fig = plt.figure()
l = []
for i,tmp in enumerate(models_gs_lat):
    tmp = tmp-np.mean(tmp[0:20])
    l_tmp = plt.plot(models_years[i],running_mean.running_mean(tmp,10),label=models[i])
    l += l_tmp

plt.xlabel('year')
plt.ylabel('NAC latitude (anomaly from 1st 20yrs)')
plt.xlim(2000,2100)
plt.legend(loc =  'upper left',prop={'size':5})
plt.savefig('/home/ph290/Documents/figures/gulfstream_analysis/gulf_stream_rcp85_more_models.png')
#plt.close("all")
plt.show()
