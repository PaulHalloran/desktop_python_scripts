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
import iris.plot as iplt
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
import scipy.signal
import iris.analysis.stats as istats


def unique_models(all_files):
    model = []
    for file in all_files:
        model.append(file.split('/')[-1].split('_')[0])
    return np.unique(np.array(model))




def butter_bandpass(lowcut,  cutoff):
    order = 2
    low = 1/lowcut
    b, a = scipy.signal.butter(order, low , btype=cutoff,analog = False)
    return b, a
 
 

 
def low_pass_filter(cube,limit_years):
        b1, a1 = butter_bandpass(limit_years, 'low')
        output = scipy.signal.filtfilt(b1, a1, cube,axis = 0)
        return output

 
 
 
def high_pass_filter(cube,limit_years):
        b1, a1 = butter_bandpass(limit_years, 'high')
        output = scipy.signal.filtfilt(b1, a1, cube,axis = 0)
        return output




directory = '/data/NAS-ph290/ph290/data_for_reynolds_v2/regridded/'
directory2 = '/media/usb_external1/cmip5/pr_regridded'
files = glob.glob(directory+'*tas*r1i1p1*.nc')

models = unique_models(files)
variables = ['tas']

###
#read in data
###


models = list(models)
models.remove('CCSM4')
models.remove('CMCC-CESM')
models.remove('CMCC-CM')
models.remove('CMCC-CMS')
models.remove('CESM1-FASTCHEM')
models.remove('HadGEM2-CC')
#models.remove('MIROC-ESM')

models = np.array(models)

upper_limit_years = 30.0
lower_limit_years = 5.0

data = {}

for i,model in enumerate(models):
	print 'testing: '+model
        file1 = glob.glob(directory+model+'_*'+'tas'+'*r1i1p1*.nc')
        file2 = glob.glob(directory+model+'_*'+'tos'+'*r1i1p1*.nc')
        file3 = glob.glob(directory+model+'_*'+'sos'+'*r1i1p1*.nc')
        if ((np.size(file1) > 0) & (np.size(file2) > 0) & (np.size(file3) > 0)):
		print 'processing: '+model
		#tas
		file = glob.glob(directory+model+'_*'+'tas'+'*r1i1p1*.nc')
		cube = iris.load_cube(file)
		coord = cube.coord('time')
		dt = coord.units.num2date(coord.points)
		years = np.array([coord.units.num2date(value).year for value in coord.points])
		try:
				cube.coord('latitude').guess_bounds()
				cube.coord('longitude').guess_bounds()
		except:
				print 'already have bounds'		
		grid_areas = iris.analysis.cartography.area_weights(cube)
		ts = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
		ts_filtered = ts.copy()
		ts_filtered.data = high_pass_filter(ts_filtered.data,upper_limit_years)
		ts_filtered.data = low_pass_filter(ts_filtered.data,lower_limit_years)	
		#density
		file = glob.glob(directory+model+'_*'+'tos'+'*r1i1p1*.nc')
		cube1 = iris.load_cube(file)
		file = glob.glob(directory+model+'_*'+'sos'+'*r1i1p1*.nc')
		cube2 = iris.load_cube(file)
		coord = cube1.coord('time')
		dt = coord.units.num2date(coord.points)
		years1 = np.array([coord.units.num2date(value).year for value in coord.points])
		locs = np.in1d(years, years1)
		cube1 = cube1[locs]
		cube2 = cube2[locs]
		ts_filtered = ts_filtered[locs]
		cube = cube1.copy()
		cube.data = seawater.dens(cube2.data,cube1.data-273.15)
		cube.data = high_pass_filter(cube.data,upper_limit_years)
		cube.data = low_pass_filter(cube.data,lower_limit_years)
		ts_filtered_2D = cube.copy()
		ts_filtered_2D.data = np.swapaxes(np.swapaxes(np.tile(ts_filtered.data,[180,360,1]),1,2),0,1)
		data[model] = {}
		data[model]['density'] = {}
		data[model]['density'] = istats.pearsonr(ts_filtered_2D,cube,corr_coords=['time'])
		cube1.data = high_pass_filter(cube1.data,upper_limit_years)
		cube1.data = low_pass_filter(cube1.data,lower_limit_years)
		data[model]['tos'] = {}
		data[model]['tos'] = istats.pearsonr(ts_filtered_2D,cube1,corr_coords=['time'])



plt.close('all')
plt.subplot(4, 4, 1)

for i,model in enumerate(data.viewkeys()):
	plt.subplot(4, 4, i)
	iplt.contourf(data[model]['density'],np.linspace(-1,1,21))
	plt.gca().coastlines()
	plt.title(model)

plt.savefig('/home/ph290/Documents/figures/density_correlations.png')
#plt.show(block = False)


plt.close('all')
plt.subplot(4, 4, 1)

for i,model in enumerate(data.viewkeys()):
	plt.subplot(4, 4, i)
	iplt.contourf(data[model]['tos'],np.linspace(-1,1,21))
	plt.gca().coastlines()
	plt.title(model)

plt.savefig('/home/ph290/Documents/figures/tos_correlations.png')
#plt.show(block = False)





