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
    

def extract_years(cube):
	try:
		iris.coord_categorisation.add_year(cube, 'time', name='year2')
	except:
		'already has year2'
	loc = np.where((cube.coord('year2').points >= start_year) & (cube.coord('year2').points <= end_year))
	loc2 = cube.coord('time').points[loc[0][-1]]
	cube = cube.extract(iris.Constraint(time = lambda time_tmp: time_tmp <= loc2))
	return cube


end_date = end_year = 1850
start_date = start_year = 850
expected_years = np.arange(850,1850)

import string
names = list(string.ascii_lowercase)

west = -24
east = -13
south = 65
north = 67

models = ['MRI-CGCM3', 'bcc-csm1-1', 'MPI-ESM-P', 'GISS-E2-R', 'CSIRO-Mk3L-1-2', 'HadCM3', 'MIROC-ESM', 'CCSM4']
# smoothing_val = 10
b, a = butter_lowpass(1.0/100.0, 1.0,2)
b2, a2 = butter_highpass(1.0/10.0, 1.0,2)
directory = '/data/NAS-ph290/ph290/cmip5/last1000/'

offsets = np.array([-20,-15,-10,-5,0,5,10,15,20])
for j,offset in enumerate(offsets):
	models_tmp = []
	composite_data = {}
	corr_variable = 'sos'
	for model in models:
		print model
		cube1 = iris.load_cube(directory+model+'_sos_past1000_r1i1p1_*.nc')
		cube1 = extract_years(cube1)
		if str(cube1.coord(dimensions=1).standard_name) == 'depth':
			cube1 = cube1.extract(iris.Constraint(depth = 0))
		cube1b = cube1.copy()
		cube1b.data = scipy.signal.filtfilt(b, a, cube1.data,axis = 0)
		cube1b.data = scipy.signal.filtfilt(b2, a2,cube1b.data ,axis = 0)
		cube1b.data = np.ma.masked_array(cube1b.data)
		cube1b.data.mask = cube1.data.mask
		cube1b = extract_years(cube1b)
		cube2 = cube1b.intersection(longitude = (west, east))
		cube2 = cube2.intersection(latitude = (south, north))
		try:
			cube2.coord('latitude').guess_bounds()
			cube2.coord('longitude').guess_bounds()
		except:
					'has bounds'
		grid_areas = iris.analysis.cartography.area_weights(cube2)
		ts_variable = cube2.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas).data
		#ts_variable = rm.running_mean(ts_variable,smoothing_val)
		min_loc = np.where(ts_variable <= np.mean(ts_variable) - np.std(ts_variable))
		max_loc = np.where(ts_variable >= np.mean(ts_variable) + np.std(ts_variable))
		coord = cube1.coord('time')
		dt = coord.units.num2date(coord.points)
		year_corr_variable = np.array([coord.units.num2date(value).year for value in coord.points])       
		loc_min2 = np.in1d(year_corr_variable,year_corr_variable[min_loc]+offset)
		loc_max2 = np.in1d(year_corr_variable,year_corr_variable[max_loc]+offset)
		composite_data[model] = {}
		composite_data[model]['composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
		composite_data[model]['composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
		composite_data[model]['composite_high_minus_low'] = composite_data[model]['composite_high'] - composite_data[model]['composite_low']
		models_tmp.append(model)
		
	composite_mean_low = composite_data[models_tmp[0]]['composite_high'].copy()
	composite_mean_high = composite_data[models_tmp[0]]['composite_high'].copy()
	composite_mean_data_low = composite_mean_low.data.copy() * 0.0
	composite_mean_data_high = composite_mean_low.data.copy() * 0.0

	i = 0
	for model in models_tmp:
			i += 1
			print model
			composite_mean_data_low += composite_data[model]['composite_low'].data
			composite_mean_data_high += composite_data[model]['composite_high'].data


	composite_mean_low.data = composite_mean_data_low
	composite_mean_low = composite_mean_low / i
	composite_mean_high.data = composite_mean_data_high
	composite_mean_high = composite_mean_high / i

	min1 = np.min(composite_mean_high.data)
	min2 = np.min(composite_mean_low.data)
	min = np.min([min1,min2])
	max1 = np.max(composite_mean_high.data)
	max2 = np.max(composite_mean_low.data)
	max = np.max([max1,max2])
	max2 = np.max([max,np.abs(min)])

	min_use = -0.2
	max_use = 0.2

	tmp_high = composite_mean_high.copy()
	tmp_low = composite_mean_low.copy()

	plt.close('all')
	qplt.contourf(tmp_high,np.linspace(min_use,max_use),cmap='bwr')
	plt.gca().coastlines()
	plt.title('PMIP3 high salinity yr '+corr_variable+' composites, n = '+str(i))
	plt.savefig('/home/ph290/Documents/figures/sainity_composites_'+names[j]+'.png')
	#plt.show()

'''

    for model in models_tmp:
            if np.size(np.shape(composite_data[model]['composite_high'])) == 3:
                    composite_data[model]['composite_high'] = composite_data[model]['composite_high'][0]
                    composite_data[model]['composite_low'] = composite_data[model]['composite_low'][0]


    for model in models_tmp:
         if not(corr_variable == 'msftbarot'):
            print model
            c1 = composite_data[model]['composite_low']
            c2 = composite_data[model]['composite_high']
            composite_data[model]['composite_low'].data = np.ma.masked_where(c1.data < -10000,c1.data)
            composite_data[model]['composite_high'].data = np.ma.masked_where(c2.data < -10000,c2.data)
            composite_data[model]['composite_low'].data = np.ma.masked_where(c1.data > 10000,c1.data)
            composite_data[model]['composite_high'].data = np.ma.masked_where(c2.data > 10000,c2.data)


    composite_mean_low = composite_data[models_tmp[0]]['composite_high'].copy()
    composite_mean_high = composite_data[models_tmp[0]]['composite_high'].copy()
    composite_mean_data_low = composite_mean_low.data.copy() * 0.0
    composite_mean_data_high = composite_mean_low.data.copy() * 0.0

    i = 0
    for model in models_tmp:
            i += 1
            print model
            composite_mean_data_low += composite_data[model]['composite_low'].data
            composite_mean_data_high += composite_data[model]['composite_high'].data


    composite_mean_low.data = composite_mean_data_low
    composite_mean_low = composite_mean_low / i
    composite_mean_high.data = composite_mean_data_high
    composite_mean_high = composite_mean_high / i


    min1 = np.min(composite_mean_high.data)
    min2 = np.min(composite_mean_low.data)
    min = np.min([min1,min2])
    max1 = np.max(composite_mean_high.data)
    max2 = np.max(composite_mean_low.data)
    max = np.max([max1,max2])
    max2 = np.max([max,np.abs(min)])

    if min < 0:
            min_use = min
            max_use = max



    if min >= 0:
            min_use = max2*(-1.0)
            max_use = max2	

    min_use = -0.05
    max_use = 0.05

    tmp_high = composite_mean_high.copy()
    tmp_low = composite_mean_low.copy()

    plt.close('all')
    fig = plt.figure(figsize = (10,10))
    ax1 = plt.subplot(111,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
    ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
    my_plot = iplt.contourf(tmp_high,np.linspace(min_use,max_use),cmap='bwr')
    #ax1.add_feature(cfeature.LAND,facecolor='#f6f6f6')
    plt.gca().coastlines()
    bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
    bar.set_label(corr_variable+' ('+format(tmp_high.units)+')')
    plt.title('PMIP3 high salinity yr '+corr_variable+' composites, n = '+str(i))
    plt.savefig('/home/ph290/Documents/figures/volc_composites_'+corr_variable+'_'+names[j]+'.png')
    #plt.show()


'''