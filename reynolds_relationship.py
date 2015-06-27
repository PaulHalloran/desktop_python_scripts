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


def unique_models(all_files):
    model = []
    for file in all_files:
        model.append(file.split('/')[-1].split('_')[0])
    return np.unique(np.array(model))



###
#Filter
###

# N=5.0
# #N is the order of the filter - i.e. quadratic
# timestep_between_values=1.0 #years value should be '1.0/12.0'
# low_cutoff=100.0
# 
# Wn_low=timestep_between_values/low_cutoff
# 
# b, a = scipy.signal.butter(N, Wn_low, btype='high')

def butter_bandpass(lowcut, fs, order=5):
    nyq = fs
    low = lowcut/nyq
    b, a = scipy.signal.butter(order, low , btype='high',analog = False)
    return b, a
   
#b, a = butter_bandpass(1.0/100.0, 1)




def high_pass(x,window_len,window='hamming'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    z = x-y[0:-1.0*(window_len-1)]
    return z


smoothing_window = 100

directory = '/data/NAS-ph290/ph290/data_for_reynolds_v2/regridded/'
files = glob.glob(directory+'*r1i1p1*.nc')

models = unique_models(files)
variables = ['tas','tos','sos']

###
#read in data
###

lon_west1 = 0
lon_east1 = 360
lat_south1 = 0.0
lat_north1 = 90.0

lon_west2 = -25.0
lon_east2 = -10.0
lat_south2 = 65.0
lat_north2 = 70.0

data = {}

models = list(models)
models.remove('CCSM4')
models.remove('CMCC-CESM')
models.remove('CMCC-CM')
models.remove('CMCC-CMS')
models.remove('CESM1-FASTCHEM')
models = np.array(models)

for i,model in enumerate(models):
	print 'testing: '+model
        file1 = glob.glob(directory+model+'_*'+variables[0]+'*r1i1p1*.nc')
        file2 = glob.glob(directory+model+'_*'+variables[1]+'*r1i1p1*.nc')
        file3 = glob.glob(directory+model+'_*'+variables[2]+'*r1i1p1*.nc')
        if ((np.size(file1) > 0) & (np.size(file2) > 0) & (np.size(file3) > 0)):
            print 'processing: '+model
            data[model] = {}
            for variable in variables:
                file = glob.glob(directory+model+'_*'+variable+'*r1i1p1*.nc')
                cube = iris.load_cube(file)
                lon_west = lon_west2
                lon_east = lon_east2
                lat_south = lat_south2
                lat_north = lat_north2
                if variable == 'tas':
                        lon_west = lon_west1
                        lon_east = lon_east1
                        lat_south = lat_south1
                        lat_north = lat_north1
                try:
                        cube.coord('latitude').guess_bounds()
                        cube.coord('longitude').guess_bounds()
                except:
                        print 'already have bounds'
                cube = cube.intersection(longitude=(lon_west, lon_east))
                cube = cube.intersection(latitude=(lat_south, lat_north))
                #cube.coord('latitude').guess_bounds()
                #cube.coord('longitude').guess_bounds()
                grid_areas = iris.analysis.cartography.area_weights(cube)
                ts = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
                data[model][variable] = ts.data
                coord = cube.coord('time')
                dt = coord.units.num2date(coord.points)
                years = np.array([coord.units.num2date(value).year for value in coord.points])
                if variable == 'tas':
                    data[model]['tas_year'] = years
                if variable == 'tos':
                    data[model]['tos_year'] = years
                if variable == 'sos':
                    data[model]['sos_year'] = years


for i,model in enumerate(models):
    	print 'testing: '+model
        file1 = glob.glob(directory+model+'_*'+variables[0]+'*r1i1p1*.nc')
        file2 = glob.glob(directory+model+'_*'+variables[1]+'*r1i1p1*.nc')
        file3 = glob.glob(directory+model+'_*'+variables[2]+'*r1i1p1*.nc')
        if ((np.size(file1) > 0) & (np.size(file2) > 0) & (np.size(file3) > 0)):
            print 'processing: '+model
            size = np.size(np.where(data[model]['tos_year']))
            data[model]['density'] = np.zeros(size)
            data[model]['density'][:] = np.NAN
	    if np.size(data[model]['sos']) == np.size(data[model]['tos']):
            	for yr in data[model]['tas_year']:
                	loc = np.where(data[model]['tos_year'] == yr)
                	if np.size(loc) > 0:
                            if np.mean(data[model]['sos'].data) < 10.0:
                    		data[model]['density'][loc] = seawater.dens(data[model]['sos'][loc]*1000.0,data[model]['tos'][loc]-273.15)
                            if np.mean(data[model]['sos'].data) > 10.0:
                    		data[model]['density'][loc] = seawater.dens(data[model]['sos'][loc],data[model]['tos'][loc]-273.15)


for i,model in enumerate(models):
    	print 'testing: '+model
        file1 = glob.glob(directory+model+'_*'+variables[0]+'*r1i1p1*.nc')
        file2 = glob.glob(directory+model+'_*'+variables[1]+'*r1i1p1*.nc')
        file3 = glob.glob(directory+model+'_*'+variables[2]+'*r1i1p1*.nc')
        if ((np.size(file1) > 0) & (np.size(file2) > 0) & (np.size(file3) > 0)):
            plt.plot(data[model]['tas'].data - np.mean(data[model]['tas'].data),'r')
            plt.plot(data[model]['density'] - scipy.stats.nanmean(data[model]['density']),'b')
            plt.title(model)
            plt.show()

'''

for i,model in enumerate(models):
    	print 'testing: '+model
        file1 = glob.glob(directory+model+'_*'+variables[0]+'*r1i1p1*.nc')
        file2 = glob.glob(directory+model+'_*'+variables[1]+'*r1i1p1*.nc')
        file3 = glob.glob(directory+model+'_*'+variables[2]+'*r1i1p1*.nc')
        if ((np.size(file1) > 0) & (np.size(file2) > 0) & (np.size(file3) > 0)):
            f = open('/data/NAS-ph290/ph290/data_for_reynolds_v2/'+model+'.txt','w')
            for i,dummy in enumerate(data[model]['tas_year']):
                yr = str(data[model]['tas_year'][i])
                dens = str(data[model]['density'][i])
                temp = str(data[model]['tas'][i])
                f.write(yr+','+dens+','+temp+'\n')
            f.close()
'''
