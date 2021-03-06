

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
from eofs.iris import Eof
import cartopy.feature as cfeature
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from scipy.stats import gaussian_kde
from statsmodels.stats.outliers_influence import summary_table
import iris.analysis.stats


###
#Filter
###


def butter_bandpass(lowcut, fs, order=5):
    nyq = fs
    low = lowcut/nyq
    b, a = scipy.signal.butter(order, low , btype='high',analog = False)
    return b, a
    
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
b2, a2 = butter_highpass(1.0/3, 1.0,2)
b3, a3 = butter_highpass(1.0/10, 1.0,2)

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
smoothing_val = 20
 


west = -24
east = -5
south = 65
north = 81

west = -50
east = 0
south = 45
north = 81

west = -24
east = -13
south = 65
north = 70

# west = -24
# east = -14
# south = 65
# north = 75


#models = ['MRI-CGCM3', 'bcc-csm1-1', 'MPI-ESM-P', 'GISS-E2-R', 'CSIRO-Mk3L-1-2', 'HadCM3', 'MIROC-ESM', 'CCSM4']

models = ['MRI-CGCM3', 'MPI-ESM-P', 'GISS-E2-R','CSIRO-Mk3L-1-2', 'HadCM3', 'MIROC-ESM', 'CCSM4']

directory = '/data/NAS-ph290/ph290/cmip5/last1000/'
data_and_forcings = {}

for model in models:
	print model
	data_and_forcings[model] = {}
	##############
	#  evaporation
	##############
	cube1_evap = iris.load_cube(directory+model+'_evspsbl_past1000_r1i1p1_*.nc')
	cube1_evap = extract_years(cube1_evap)
	cube1b = cube1_evap.intersection(longitude = (west, east))
	cube1b = cube1b.intersection(latitude = (south, north))
	try:
		cube1b.coord('latitude').guess_bounds()
		cube1b.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube1b)
	cube1b = cube1b.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	data_and_forcings[model]['evap_n_iceland'] = cube1b.data
	##############
	#  precip.
	##############
	cube1_precip = iris.load_cube(directory+model+'_pr_past1000_r1i1p1_*.nc')
	cube1_precip = extract_years(cube1_precip)
	cube1c = cube1_precip.intersection(longitude = (west, east))
	cube1c = cube1c.intersection(latitude = (south, north))
	try:
		cube1c.coord('latitude').guess_bounds()
		cube1c.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube1c)
	cube1c = cube1c.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	data_and_forcings[model]['pr_n_iceland'] = cube1c.data
	##############
	#  p-e
	##############
	cube1_e_min_p = cube1_precip - cube1_evap
	cube1d = cube1_e_min_p.intersection(longitude = (west, east))
	cube1d = cube1d.intersection(latitude = (south, north))
	try:
		cube1d.coord('latitude').guess_bounds()
		cube1d.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube1d)
	cube1d = cube1d.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	data_and_forcings[model]['p_minus_e_n_iceland'] = cube1d.data
	##############
	#  salinity
	##############
	#1) does model have surface-level salinity?
	try:
		cube1 = iris.load_cube(directory+model+'_sos_past1000_r1i1p1_*.nc')
                if np.size(np.shape(cube1)) == 4:
                    cube1 = cube1[:,0,:,:]
		cube1 = extract_years(cube1)
	except:
		#3) does model have 3D salinity (so)?
                cube1 = iris.load_cube(directory+model+'_so_past1000_r1i1p1_*.nc')
                cube1 = cube1.extract(iris.Constraint(str(cube1.coord(dimensions=1).standard_name)+' = '+str(np.min(cube1.coord(dimensions=1).points))))
                cube1 = extract_years(cube1)
	cube1 = cube1.intersection(longitude = (west, east))
	cube1 = cube1.intersection(latitude = (south, north))
	try:
		cube1.coord('latitude').guess_bounds()
		cube1.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube1)
	cube1 = cube1.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	data_and_forcings[model]['sos_n_iceland'] = cube1.data
	##############
	#  Spatial correlation
	##############
	#P-e
	ts = data_and_forcings[model]['sos_n_iceland']
	ts = scipy.signal.filtfilt(b1, a1, ts)
	ts = scipy.signal.filtfilt(b2, a2, ts)
	cube = cube1_e_min_p.copy()
	cube.data = scipy.signal.filtfilt(b1, a1, cube.data,axis = 0)
	cube.data = scipy.signal.filtfilt(b2, a2, cube.data,axis = 0)
	cube = iris.analysis.maths.multiply(cube, 0.0)
	ts2 = np.swapaxes(np.swapaxes(np.tile(ts,[180,360,1]),1,2),0,1)
	cube = iris.analysis.maths.add(cube, ts2)
	data_and_forcings[model]['p_minus_e_s_corr'] =  iris.analysis.stats.pearsonr(cube1_e_min_p, cube, corr_coords=['time'])
	#P
	ts = data_and_forcings[model]['sos_n_iceland']
	ts = scipy.signal.filtfilt(b1, a1, ts)
	ts = scipy.signal.filtfilt(b2, a2, ts)
	cube = cube1_precip.copy()
	cube.data = scipy.signal.filtfilt(b1, a1, cube.data,axis = 0)
	cube.data = scipy.signal.filtfilt(b2, a2, cube.data,axis = 0)
	cube = iris.analysis.maths.multiply(cube, 0.0)
	ts2 = np.swapaxes(np.swapaxes(np.tile(ts,[180,360,1]),1,2),0,1)
	cube = iris.analysis.maths.add(cube, ts2)
	data_and_forcings[model]['p_s_corr'] =  iris.analysis.stats.pearsonr(cube1_precip, cube, corr_coords=['time'])
	#E
	ts = data_and_forcings[model]['sos_n_iceland']
	ts = scipy.signal.filtfilt(b1, a1, ts)
	ts = scipy.signal.filtfilt(b2, a2, ts)
	cube = cube1_evap.copy()
	cube.data = scipy.signal.filtfilt(b1, a1, cube.data,axis = 0)
	cube.data = scipy.signal.filtfilt(b2, a2, cube.data,axis = 0)
	cube = iris.analysis.maths.multiply(cube, 0.0)
	ts2 = np.swapaxes(np.swapaxes(np.tile(ts,[180,360,1]),1,2),0,1)
	cube = iris.analysis.maths.add(cube, ts2)
	data_and_forcings[model]['e_s_corr'] =  iris.analysis.stats.pearsonr(cube1_evap, cube, corr_coords=['time'])
	##############
	#  time
	##############
	coord = cube1.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	data_and_forcings[model]['year'] = year  



##############
#  average spatial data together
##############

#P-E
shape = np.shape(data_and_forcings[models[0]]['p_minus_e_s_corr'])
size = np.size(models)
data_array = np.zeros([size,shape[0],shape[1]]) + np.NAN
for i,model in enumerate(models):
	data_array[i,:,:] = data_and_forcings[models[0]]['p_minus_e_s_corr'].data


p_minus_e_s_corr_cube = data_and_forcings[models[0]]['p_minus_e_s_corr'].copy()
p_minus_e_s_corr_cube.data = scipy.stats.nanmean(data_array,axis = 0)


#P
shape = np.shape(data_and_forcings[models[0]]['p_s_corr'])
size = np.size(models)
data_array = np.zeros([size,shape[0],shape[1]]) + np.NAN
for i,model in enumerate(models):
	data_array[i,:,:] = data_and_forcings[models[0]]['p_s_corr'].data


p_s_corr_cube = data_and_forcings[models[0]]['p_s_corr'].copy()
p_s_corr_cube.data = scipy.stats.nanmean(data_array,axis = 0)

#E
shape = np.shape(data_and_forcings[models[0]]['e_s_corr'])
size = np.size(models)
data_array = np.zeros([size,shape[0],shape[1]]) + np.NAN
for i,model in enumerate(models):
	data_array[i,:,:] = data_and_forcings[models[0]]['e_s_corr'].data


e_s_corr_cube = data_and_forcings[models[0]]['e_s_corr'].copy()
e_s_corr_cube.data = scipy.stats.nanmean(data_array,axis = 0)

###
# plotting
###

fig = plt.figure(figsize=(5, 14))

ax1 = plt.subplot(311)
qplt.contourf(p_minus_e_s_corr_cube,31)
plt.title('P-E corr')
plt.gca().coastlines()

ax2 = plt.subplot(312)
qplt.contourf(p_s_corr_cube,31)
plt.title('P corr')
plt.gca().coastlines()

ax3 = plt.subplot(313)
qplt.contourf(e_s_corr_cube,31)
plt.title('E corr')
plt.gca().coastlines()

plt.show()

##############
#  average models together
##############

start_date = 850
end_date = 1850
tmp_years = np.arange(start_date,end_date+1)

def average_accross_models(tmp_years,data_and_forcings,models,variable):
	out = np.empty([np.size(models),np.size(tmp_years)]) * 0.0 + np.NAN
	for i,model in enumerate(models):
		data = data_and_forcings[model][variable]
		data = (data-np.min(data))/(np.max(data)-np.min(data))
		for j,yr in enumerate(tmp_years):
			loc = np.where(data_and_forcings[model]['year'] == yr)
			if np.size(loc) != 0:
				out[i,j] = np.mean(data[loc])
		out[i,:] = scipy.signal.filtfilt(b1, a1, out[i,:])
		out[i,:] = scipy.signal.filtfilt(b2, a2, out[i,:])
	return scipy.stats.nanmean(out,axis = 0)



sos_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'sos_n_iceland')
evap_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'evap_n_iceland')
pr_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'pr_n_iceland')
p_minus_e_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'p_minus_e_n_iceland')


y = sos_n_iceland_mean
y = y - np.mean(sos_n_iceland_mean)
plt.plot(y,'b')
y = p_minus_e_n_iceland_mean * (-1.0)
#y = evap_n_iceland_mean
y = y - np.mean(y)
plt.plot(y,'r')
plt.show()

'''
OK, so need to find what area's p-e correlates with the N. iceland salinity variabnlity - it is likely to be the area contributing to it - then compare tome sereies...
'''
