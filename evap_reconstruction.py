

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

###########################################################################################################
#                                                                                                         #
#   start of non stream function bit (stream function stuff done in palaeo_amo_amoc_paper_figures_iv.py)  #
#                                                                                                         #
###########################################################################################################

end_date = end_year = 1850
start_date = start_year = 850
expected_years = np.arange(850,1850)
smoothing_val = 20
 


#west = -24
#east = -5
#south = 65
#north = 81


west = -24
east = -13
south = 65
north = 67
'''
'''
# west = -24
# east = -14
# south = 65
# north = 75


#models = ['MRI-CGCM3', 'bcc-csm1-1', 'MPI-ESM-P', 'GISS-E2-R', 'CSIRO-Mk3L-1-2', 'HadCM3', 'MIROC-ESM', 'CCSM4']

models = ['MRI-CGCM3', 'MPI-ESM-P', 'GISS-E2-R','CSIRO-Mk3L-1-2', 'HadCM3', 'MIROC-ESM', 'CCSM4']

directory = '/data/NAS-ph290/ph290/cmip5/last1000/monthly/'
data_and_forcings = {}

for model in models:
	print model
	data_and_forcings[model] = {}
		###########################
		#   Whole Arctic sea ice  #
		###########################
	cube1 = iris.load_cube(directory+model+'_sic_past1000_r1i1p1_*.nc')
	cube1 = extract_years(cube1)
	temporary_cube = cube1.intersection(longitude = (-180, 180))
	arctic_sea_ice_fraction = temporary_cube.intersection(latitude = (0, 90))
	try:
		arctic_sea_ice_fraction.coord('latitude').guess_bounds()
		arctic_sea_ice_fraction.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(arctic_sea_ice_fraction)
	arctic_sea_ice_area = arctic_sea_ice_fraction.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
		###########################
		#   N Iceland sea ice     #
		###########################
	data_and_forcings[model]['ice_area'] = arctic_sea_ice_area.data
	temporary_cube = cube1.intersection(longitude = (west, east))
	sea_ice_fraction = temporary_cube.intersection(latitude = (south, north))
	try:
		sea_ice_fraction.coord('latitude').guess_bounds()
		sea_ice_fraction.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(sea_ice_fraction)
	sea_ice_area = sea_ice_fraction.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	data_and_forcings[model]['ice_area_n_iceland_monthly'] = sea_ice_area.data
		###########################
		#   N Iceland wind speed  #
		###########################
	#1) does model have raw near-surface wind-speeds (sfcWind)?
	try:
		cube1 = iris.load_cube(directory+model+'_sfcWind_past1000_r1i1p1_*.nc')
		cube1 = extract_years(cube1)
	except:
		#2) does model have near surface u and v winds (uas, vas)?
		try:
			cube1 = extract_years(iris.load_cube(directory+model+'_uas_past1000_r1i1p1*.nc'))
			cube1_b = extract_years(iris.load_cube(directory+model+'_vas_past1000_r1i1p1*.nc'))
			cube1 = ((cube1*cube1) + (cube1_b*cube1_b))**(.5)
		except:
			#3) does model have 3D winds (ua and va)?
			cube1 = iris.load_cube(directory+model+'_ua_past1000_r1i1p1_*.nc')
			#extracting bottom two pressure levels and combining to give a complete map, because surface level was too high pressure for N. Iceland.
			cube1_l1 = cube1[:,0,:,:]
			cube1_l1.data.data[np.where(cube1_l1.data.data > 1.0e10)] = 0.0
			cube1_l2 = cube1[:,1,:,:]
			cube1_l2.data.data[np.where(cube1_l2.data.data > 1.0e10)] = 0.0
			mask = cube1_l1.data.mask
			mask =  np.invert(mask)
			cube1_l2.data.mask = mask
			cube1_surface = cube1_l1.copy()
			cube1_surface.data = cube1_l1.data.data + cube1_l2.data.data
			cube1 = cube1_surface
			cube1 = extract_years(cube1)
			cube1_b = extract_years(iris.load_cube(directory+model+'_va_past1000_r1i1p1_*.nc'))
			#extracting bottom two pressure levels and combining to give a complete map, because surface level was too high pressure for N. Iceland.
			cube1_l1 = cube1b[:,0,:,:]
			cube1_l1.data.data[np.where(cube1_l1.data.data > 1.0e10)] = 0.0
			cube1_l2 = cube1b[:,1,:,:]
			cube1_l2.data.data[np.where(cube1_l2.data.data > 1.0e10)] = 0.0
			mask = cube1_l1.data.mask
			mask =  np.invert(mask)
			cube1_l2.data.mask = mask
			cube1_surface = cube1_l1.copy()
			cube1_surface.data = cube1_l1.data.data + cube1_l2.data.data
			cube1b = cube1_surface
			cube1_b = extract_years(cube1_b)
			cube1 = ((cube1*cube1) + (cube1_b*cube1_b))**(.5)
	cube1 = cube1.intersection(longitude = (west, east))
	cube1 = cube1.intersection(latitude = (south, north))
	try:
		cube1.coord('latitude').guess_bounds()
		cube1.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube1)
	cube1 = cube1.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	data_and_forcings[model]['windspeed_n_iceland_monthly'] = cube1.data
		##################################
		#   N Iceland relative humidity  #
		##################################
	#1) does model have raw near-surface humidity (hurs)?
	try:
		cube1 = iris.load_cube(directory+model+'_hurs_past1000_r1i1p1_*.nc')
		cube1 = extract_years(cube1)
	except:
		#3) does model have 3D humidity (hur)?
		cube1 = iris.load_cube(directory+model+'_hur_past1000_r1i1p1_*.nc')
		#extracting bottom two pressure levels and combining to give a complete map, because surface level was too high pressure for N. Iceland.
		cube1_l1 = cube1[:,0,:,:]
		test = np.where(cube1_l1.data.data > 1.0e10)
		if np.size(test) > 10:
			cube1_l1.data.data[test] = 0.0
			cube1_l2 = cube1[:,1,:,:]
			cube1_l2.data.data[np.where(cube1_l2.data.data > 1.0e10)] = 0.0
			mask = cube1_l1.data.mask
			mask =  np.invert(mask)
			cube1_l2.data.mask = mask
			cube1_surface = cube1_l1.copy()
			cube1_surface.data = cube1_l1.data.data + cube1_l2.data.data
			cube1 = cube1_surface
			cube1 = extract_years(cube1)
		else:
			cube1 = extract_years(cube1_l1)
	cube1 = cube1.intersection(longitude = (west, east))
	cube1 = cube1.intersection(latitude = (south, north))
	try:
		cube1.coord('latitude').guess_bounds()
		cube1.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube1)
	cube1 = cube1.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	data_and_forcings[model]['humidity_n_iceland_monthly'] = cube1.data
		#########################################################
		#   N Iceland net surface downward shortwave radiation  #
		#########################################################
	cube1 = iris.load_cube(directory+model+'_rsds_past1000_r1i1p1_*.nc')
	cube1 = extract_years(cube1)
	cube1b = iris.load_cube(directory+model+'_rsus_past1000_r1i1p1_*.nc')
	cube1b = extract_years(cube1b)
	cube1 = cube1 - cube1b
	cube1 = cube1.intersection(longitude = (west, east))
	cube1 = cube1.intersection(latitude = (south, north))
	try:
		cube1.coord('latitude').guess_bounds()
		cube1.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube1)
	cube1 = cube1.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	data_and_forcings[model]['net_surface_downward_sw_n_iceland_monthly'] = cube1.data
		#####################################################
		#   N Iceland net TOA downward shortwave radiation  #
		#####################################################
	cube1 = iris.load_cube(directory+model+'_rsdt_past1000_r1i1p1_*.nc')
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
	data_and_forcings[model]['net_TOA_downward_sw_n_iceland_monthly'] = cube1.data
		################################
		#   N Iceland air temperature  #
		################################
	cube1 = iris.load_cube(directory+model+'_tas_past1000_r1i1p1_*.nc')
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
	data_and_forcings[model]['tas_n_iceland_monthly'] = cube1.data
		#time
	coord = arctic_sea_ice_area.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	data_and_forcings[model]['year_monthly'] = year      
	
	



directory2 = '/data/NAS-ph290/ph290/cmip5/last1000/'

for model in models:
	cube1 = iris.load_cube(directory2+model+'_evspsbl_past1000_r1i1p1_*.nc')
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
	data_and_forcings[model]['evap_n_iceland'] = cube1.data
		#time
	coord = cube1.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	data_and_forcings[model]['year'] = year  



######################################################################################
#     Reconstructing evaporation from the simplified version of the Penman equation  #
#     as described in Eq 32 of Valiantzas Journal of Hydrology (2006)                #
######################################################################################

######################################################################################
# evap = 0.051 * Rs * sqrt(T + 9.5) - 2.4 * (Rs/Ra) * (Rs/Ra) + 0.052 * (T + 20) * (1 - (RH/100)) * (au - 0.38 + 0.54 * u) 
# 	units: mm/d - note that cmip5 is kg/m2/s
# Where:
# evap is the potential open water evaporation
# T = air temperature oC (cmip5 = tas, units K) 
# Rs = net surface downward solar radiation (MJ/m2/d) (i.e. 1- albedo (alpha) * solar radiation) (cmip5 = rsds - rsus, or over the ocean: rsntds alone cos net, units W/m2 = J/s/m2)
# Ra = is the extraterrestrial radiation (MJ/m2/d) (i.e. TOA downwelling shortwave) (cmip5 = rsdt, units W/m2)
# RH = relative humidity (%)(cmip5 = hurs)
# au = wind function coefficient = 0.0 (although might be 0.54 for large body of open water)
# u = wind speed at 2 m height (m/s) (cmip5 vas, units m/s)
# note also that you might want to correct for sea-ice concentration
######################################################################################



X = np.arange(2,10)
Y = (X + 9.5)**(0.5)
X2 = sm.add_constant(X)
mod = sm.OLS(Y,X2)
results = mod.fit()

def penman(T,Rs,Ra,RH,u):
	au = 0.5
	es = 0.611*np.exp((17.27*T)/(T+237.3))
	P = 101.3*((293.0-(0.0065*0.0))/293.0)**(5.26)
	lamb = 2.501-(2.361e-3)*T
	delta = (4098.0*es)/((T+237.3)*(T+237.3))
	gamma = 0.0016286*(P/lamb)
	return (1.0/lamb * Rs * (delta/(delta+gamma)) - 2.4 * (Rs/Ra) * (Rs/Ra) + 0.052 * (T + 20.0) * (1.0 - (RH/100.0)) * (au - 0.38 + 0.54 * u))



for model in models:
	print 'note: models may change volcanic forcing differently such that TOA radiation does not include volcanoc reductions - a further variable exists in CMIP5 if required'
	day_sec = 60.0 * 60.0 * 24.0
	#convert mm/day = kg/m2/s
	T = data_and_forcings[model]['tas_n_iceland_monthly'] - 273.15
	Rs = data_and_forcings[model]['net_surface_downward_sw_n_iceland_monthly'] * day_sec * 1.9e-6
	Ra = data_and_forcings[model]['net_TOA_downward_sw_n_iceland_monthly'] * day_sec * 1.9e-6
	RH = data_and_forcings[model]['humidity_n_iceland_monthly']
	u = data_and_forcings[model]['windspeed_n_iceland_monthly']
	#note that the part of the relationship '(T + 9.5)**(0.5)' breaks down with negative temperatures (because you can't sqrt negatives)
	#so I've gone back to the full calculation
	sea_ice_fraction = data_and_forcings[model]['ice_area_n_iceland_monthly']
	data_and_forcings[model]['reconstructed_evap_monthly'] = penman(T,Rs,Ra,RH,u) / day_sec
	data_and_forcings[model]['reconstructed_evap_multiplied_by_seaice_monthly'] = data_and_forcings[model]['reconstructed_evap_monthly'] * (-1.0*(sea_ice_fraction/100.0))
	T_temp = T.copy()
	T_temp = T_temp * 0.0 +scipy.stats.nanmean(T)
	data_and_forcings[model]['reconstructed_evap_T_const_monthly'] = penman(T_temp,Rs,Ra,RH,u) / day_sec
	u_temp = u.copy()
	u_temp = u_temp * 0.0 +scipy.stats.nanmean(u)
	data_and_forcings[model]['reconstructed_evap_wind_const_monthly'] = penman(T,Rs,Ra,RH,u_temp) / day_sec
	RH_temp = RH.copy()
	RH_temp = RH_temp * 0.0 +scipy.stats.nanmean(RH)
	data_and_forcings[model]['reconstructed_evap_humidity_const_monthly'] = penman(T,Rs,Ra,RH_temp,u) / day_sec
	Rs_temp = Rs.copy()
	Rs_temp = Rs_temp * 0.0 +scipy.stats.nanmean(Rs)
	data_and_forcings[model]['reconstructed_evap_surface_radiation_const_monthly'] = penman(T,Rs_temp,Ra,RH,u) / day_sec
	Ra_temp = Ra.copy()
	Ra_temp = Ra_temp * 0.0 +scipy.stats.nanmean(Ra)
	data_and_forcings[model]['reconstructed_evap_TOA_radiation_const_monthly'] = penman(T,Rs,Ra_temp,RH,u) / day_sec
	T_temp = T.copy()
	T_temp = T_temp * 0.0 +scipy.stats.nanmean(T)
	Rs_temp = Rs.copy()
	Rs_temp = Rs_temp * 0.0 +scipy.stats.nanmean(Rs)
	data_and_forcings[model]['reconstructed_evap_TandRs_const_monthly'] = penman(T_temp,Rs_temp,Ra,RH,u) / day_sec


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
		#data = (data-np.min(data))/(np.max(data)-np.min(data))
		for j,yr in enumerate(tmp_years):
			loc = np.where(data_and_forcings[model]['year_monthly'] == yr)
			if np.size(loc) != 0:
				out[i,j] = np.mean(data[loc])
		out[i,:] = scipy.signal.filtfilt(b1, a1, out[i,:])
		out[i,:] = scipy.signal.filtfilt(b2, a2, out[i,:])
	return scipy.stats.nanmean(out,axis = 0)



def average_accross_models_not_monthly(tmp_years,data_and_forcings,models,variable):
	out = np.empty([np.size(models),np.size(tmp_years)]) * 0.0 + np.NAN
	for i,model in enumerate(models):
		data = data_and_forcings[model][variable]
		#data = (data-np.min(data))/(np.max(data)-np.min(data))
		for j,yr in enumerate(tmp_years):
			loc = np.where(data_and_forcings[model]['year'] == yr)
			if np.size(loc) != 0:
				out[i,j] = np.mean(data[loc])
		out[i,:] = scipy.signal.filtfilt(b1, a1, out[i,:])
		out[i,:] = scipy.signal.filtfilt(b2, a2, out[i,:])
	return scipy.stats.nanmean(out,axis = 0)



def average_accross_models2(tmp_years,data_and_forcings,models,variable):
    out = np.empty([np.size(models),np.size(tmp_years)]) * 0.0 + np.NAN
    for i,model in enumerate(models):
        data = data_and_forcings[model][variable]
        #data = scipy.signal.filtfilt(b1, a1, data)
        #data = scipy.signal.filtfilt(b2, a2, data)
        #data = (data-np.min(data))/(np.max(data)-np.min(data))
        for j,yr in enumerate(tmp_years):
            loc = np.where(data_and_forcings[model]['year_monthly'] == yr)
            if np.size(loc) != 0:
                out[i,j] = np.mean(data[loc])
    return scipy.stats.nanmean(out,axis = 0)


#import pickle
#with open('/home/ph290/Documents/python_scripts/pickles/evap.pickle', 'w') as f:
#    pickle.dump([data_and_forcings,models,tmp_years], f)
 
# with open('/home/ph290/Documents/python_scripts/pickles/evap.pickle', 'r') as f:
# 	[data_and_forcings,models,tmp_years] = pickle.load(f)

'''
evap_n_iceland_mean = average_accross_models_not_monthly(tmp_years,data_and_forcings,models,'evap_n_iceland')
reconstructed_evap_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_monthly')
reconstructed_evap_multiplied_by_seaice_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_multiplied_by_seaice_monthly')


reconstructed_evap_multiplied_by_seaice_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_multiplied_by_seaice_monthly')
ice_area_n_iceland_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'ice_area_n_iceland_monthly')
sw_n_iceland_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'net_surface_downward_sw_n_iceland_monthly')
tas_n_iceland_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'tas_n_iceland_monthly')
evap_n_iceland_mean = average_accross_models_not_monthly(tmp_years,data_and_forcings,models,'evap_n_iceland')
# evap_n_iceland_mean_not_normal_monthly = average_accross_models(tmp_years,data_and_forcings,models,'evap_n_iceland_monthly')
reconstructed_evap_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_monthly')
reconstructed_evap_T_const_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_T_const_monthly')
reconstructed_evap_surface_radiation_const_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_surface_radiation_const_monthly')
reconstructed_evap_TandRs_const_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_TandRs_const_monthly')
reconstructed_evap_multiplied_by_seaice_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_multiplied_by_seaice_monthly')
reconstructed_evap_wind_const_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_wind_const_monthly')
reconstructed_evap_humidity_const_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_humidity_const_monthly')
reconstructed_evap_surface_radiation_const_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_surface_radiation_const_monthly')
reconstructed_evap_TOA_radiation_const_mean_monthly = average_accross_models(tmp_years,data_and_forcings,models,'reconstructed_evap_TOA_radiation_const_monthly')


data1 = evap_n_iceland_mean * day_sec
#data1 = scipy.signal.filtfilt(b1, a1, data1)
#data1 = scipy.signal.filtfilt(b2, a2, data1)
#data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
data2 = reconstructed_evap_mean_monthly * day_sec
data3 = reconstructed_evap_T_const_mean_monthly * day_sec
data4 = reconstructed_evap_surface_radiation_const_mean_monthly * day_sec
# data2 = reconstructed_evap_multiplied_by_seaice_mean_monthly
#data2 = scipy.signal.filtfilt(b1, a1, data2)
#data2 = scipy.signal.filtfilt(b2, a2, data2)
#data2 = (data2-np.min(data2))/(np.max(data2)-np.min(data2))
# plt.plot(data1,'b') 
# plt.plot(data2,'r')
plt.close('all')
# plt.scatter(data1,data2,color = 'b')
# plt.scatter(data1,data3,color = 'r')
# plt.scatter(data1,data4,color = 'g')
# plt.show()


plt.plot(data1,color = 'b')
plt.plot(data2,color = 'r')
plt.plot(data3,color = 'g')
plt.show()

'''