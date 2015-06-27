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
import running_mean as rm
import running_mean_post as rmp
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import iris.analysis.cartography
import numpy.ma as ma
import scipy.interpolate
import gc
import pickle
import biggus
import seawater
import cartopy.feature as cfeature
import statsmodels.api as sm
import iris.analysis.stats

###
#Filter
###


def butter_lowpass(lowcut, fs, order=5):
    nyq = fs
    low = lowcut/nyq
    b, a = scipy.signal.butter(order, low , btype='high',analog = False)
    return b, a
    

def butter_highpass(highcut, fs, order=5):
    nyq = fs
    high = highcut/nyq
    b, a = scipy.signal.butter(order, high , btype='low',analog = False)
    return b, a
    

b1, a1 = butter_lowpass(1.0/100.0, 1.0,2)
b2, a2 = butter_lowpass(10.0/100.0, 1.0,2)

def unique_models(directory,var):
	models = []
	files = glob.glob(directory+'/*_'+var+'_piControl*.nc')
	for file in files:
		models.append(file.split('/')[-1].split('_')[0])
	return np.unique(models)



tos_dir = '/data/NAS-ph290/ph290/cmip5/reynolds_data/'
sos_dir = '/data/NAS-ph290/ph290/cmip5/reynolds_data/'
uo_dir = '/media/usb_external1/cmip5/gulf_stream_analysis/regridded/'



models1 = unique_models(tos_dir,'tos')
models2 = unique_models(sos_dir,'sos')
models3 = unique_models(uo_dir,'uo')

models = np.intersect1d(models1,models2)
models = np.intersect1d(models,models3)


#main section

def time_space_correlations(cube,timeseries,cube_year,ts_year): 
	years = np.intersect1d(cube_year,ts_year)
	years = np.intersect1d(ts_year,years)
	ind1 = np.in1d(cube_year,years)
	ind2 = np.in1d(ts_year,years)
	cube1 = cube[ind1]
	ts = timeseries[ind2]
	time = iris.coords.DimCoord(range(0, np.size(ts), 1), standard_name='time', units='seconds')
	latitude = iris.coords.DimCoord(range(-90, 90, 1), standard_name='latitude', units='degrees')
	longitude = iris.coords.DimCoord(range(0, 360, 1), standard_name='longitude', units='degrees')
	new_cube = iris.cube.Cube(np.zeros((np.size(ts),180, 360), np.float32),standard_name='sea_surface_temperature', long_name='Sea Surface Temperature', var_name='tos', units='K',dim_coords_and_dims=[(time,0), (latitude, 1), (longitude, 2)])
	new_cube.data = cube1.data
	analysis_cube = new_cube.copy()
	analysis_cube = iris.analysis.maths.multiply(analysis_cube, 0.0)
	ts2 = np.swapaxes(np.swapaxes(np.tile(ts,[analysis_cube.shape[1],analysis_cube.shape[2],1]),1,2),0,1)
	analysis_cube = iris.analysis.maths.add(analysis_cube, ts2)
	return iris.analysis.stats.pearsonr(new_cube, analysis_cube, corr_coords=[str(new_cube.coord(dimensions = 0).standard_name)])



gs_west = -60
gs_east = -50
gs_south = 30
gs_north = 45

models = list(models)
#removing duplicate models from an individual modelling centre
models.remove('CMCC-CESM')
models.remove('CMCC-CMS')
models.remove('CNRM-CM5-2')
models.remove('IPSL-CM5A-LR')
models.remove('IPSL-CM5B-LR')
models = np.array(models)

data = {}

for model in models:
	print model
	data[model] = {}
	###########################
	#    eastward velocities  #
	###########################
	cube = iris.load_cube(uo_dir+model+'_uo_piControl*.nc')
	if np.size(cube.shape) == 4:
		cube = cube[:,0,:,:]
	cube = cube.intersection(longitude = (gs_west, gs_east))
	cube = cube.intersection(latitude = (gs_south, gs_north))
	max_gulf_stream_strength = np.abs(np.ma.max(np.ma.max(cube.data,axis = 1),axis = 1))
	data[model]['gulf_stream_strength'] = max_gulf_stream_strength
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	data[model]['gulf_stream_year'] = year   
	###########################
	#    tos                  #
	###########################	
	cube = iris.load_cube(tos_dir+model+'_tos_piControl*.nc')
	cube.data = scipy.signal.filtfilt(b1, a1, cube.data,axis = 0)
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tos = np.array([coord.units.num2date(value).year for value in coord.points])
	gs = scipy.signal.filtfilt(b1, a1, data[model]['gulf_stream_strength'],axis = 0)
	data[model]['annual_gs_tos_correlation']  = time_space_correlations(cube,gs,year_tos,data[model]['gulf_stream_year'])
	cube.data = scipy.signal.filtfilt(b2, a2,cube.data ,axis = 0)
	gs = scipy.signal.filtfilt(b2, a2, gs,axis = 0)
	data[model]['decadal_gs_tos_correlation']  = time_space_correlations(cube,gs,year_tos,data[model]['gulf_stream_year'])
	###########################
	#    sos                  #
	###########################	
	cube = iris.load_cube(sos_dir+model+'_sos_piControl*.nc')
	cube.data = scipy.signal.filtfilt(b1, a1, cube.data,axis = 0)
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	year_sos = np.array([coord.units.num2date(value).year for value in coord.points])
	gs = scipy.signal.filtfilt(b1, a1, data[model]['gulf_stream_strength'],axis = 0)
	data[model]['annual_gs_sos_correlation']  = time_space_correlations(cube,gs,year_sos,data[model]['gulf_stream_year'])
	cube.data = scipy.signal.filtfilt(b2, a2,cube.data ,axis = 0)
	gs = scipy.signal.filtfilt(b2, a2, gs,axis = 0)
	data[model]['decadal_gs_sos_correlation']  = time_space_correlations(cube,gs,year_sos,data[model]['gulf_stream_year'])




def mean_cubes(data,models,variable):
	cube = data[models[0]][variable].copy()
	cube_data = cube.data.copy() * 0.0
	i = 0
	for model in models:
			i += 1
			print model
			cube_data += data[model][variable].data
	cube.data = cube_data
	return cube / i


plt.close('all')
x = 'decadal_gs_tos_correlation'
qplt.contourf(mean_cubes(data,models,x),np.linspace(-0.5,0.5,31))
plt.gca().coastlines()
plt.title(x)
plt.savefig('/home/ph290/Documents/figures/'+x+'.png')

plt.close('all')
x = 'decadal_gs_sos_correlation'
qplt.contourf(mean_cubes(data,models,x),np.linspace(-0.5,0.5,31))
plt.gca().coastlines()
plt.title(x)
plt.savefig('/home/ph290/Documents/figures/'+x+'.png')

plt.close('all')
x = 'annual_gs_sos_correlation'
qplt.contourf(mean_cubes(data,models,x),np.linspace(-0.5,0.5,31))
plt.gca().coastlines()
plt.title(x)
plt.savefig('/home/ph290/Documents/figures/'+x+'.png')

plt.close('all')
x = 'annual_gs_tos_correlation'
qplt.contourf(mean_cubes(data,models,x),np.linspace(-0.5,0.5,31))
plt.gca().coastlines()
plt.title(x)
plt.savefig('/home/ph290/Documents/figures/'+x+'.png')

