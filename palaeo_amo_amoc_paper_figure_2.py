

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

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_IIII.pickle') as f:    models,max_strm_fun,max_strm_fun_26,max_strm_fun_45,model_years,mask1,files,b,a,input_file,resolution,start_date,end_date,location = pickle.load(f)

b, a = butter_bandpass(1.0/100.0, 1.0,2)

###
#Read in Mann data
###

amo_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
amo = np.genfromtxt(amo_file, skip_header = 4)
amo_yr = amo[:,0]
amo_data = amo[:,1]
loc = np.where((amo_yr <= end_date) & (amo_yr >= start_date))
amo_yr = amo_yr[loc]
amo_data = amo_data[loc]
amo_data = scipy.signal.filtfilt(b, a, amo_data)
x = amo_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
amo_data = x


###
#read in volc data
###

#Crowley
file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)
data4 = np.genfromtxt(file4)

data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]
data = np.mean(data_tmp,axis = 1)
voln_n = data1.copy()
voln_n[:,1] = data

data_tmp[:,0] = data3[:,1]
data_tmp[:,1] = data4[:,1]
data = np.mean(data_tmp,axis = 1)
voln_s = data1.copy()
voln_s[:,1] = data

data_tmp[:,0] = data2[:,1]
data_tmp[:,1] = data4[:,1]
data = np.mean(data_tmp,axis = 1)
vol_eq = data1.copy()
vol_eq[:,1] = data

data_tmp = np.zeros([data1.shape[0],4])
data_tmp[:,0] = data2[:,1]
data_tmp[:,1] = data4[:,1]
data_tmp[:,2] = data1[:,1]
data_tmp[:,3] = data3[:,1]
data = np.mean(data_tmp,axis = 1)
vol_globe = data1.copy()
vol_globe[:,1] = data

#Gao-Robock-Ammann
file = '/data/data0/ph290/misc_data/last_millenium_volcanic/IVI2TotalInjection_501-2000Version2.txt'
data = np.genfromtxt(file,skip_header = 13)
vol_globe_GRA = np.zeros([data.shape[0],2])
vol_north_GRA = np.zeros([data.shape[0],2])
vol_globe_GRA[:,0] = data[:,0]
vol_globe_GRA[:,1] = data[:,3]
vol_north_GRA[:,0] = data[:,0]
vol_north_GRA[:,1] = data[:,1]

#crowley = vol_globe
crowley = voln_n
#GRA = vol_globe_GRA
GRA = vol_north_GRA

###
#read in solar data
###

file = '/data/data0/ph290/misc_data/last_millenium_solar/tsi_SBF_11yr.txt'
SBF_solar_in = np.genfromtxt(file,skip_header = 4)
file = '/data/data0/ph290/misc_data/last_millenium_solar/tsi_VK.txt'
VSK_solar_in = np.genfromtxt(file,skip_header = 4)
file = '/data/data0/ph290/misc_data/last_millenium_solar/tsi_DB_lin_40_11yr.txt'
DB_solar_with_back = np.genfromtxt(file,skip_header = 4, usecols=(0, 1))
DB_solar_no_back = np.genfromtxt(file,skip_header = 4, usecols=(0, 2))
#Note WSL extends to present-day rather, so often used to update end of others...
#so where says +WLS or +WLS Back, ognore here
file = '/data/data0/ph290/misc_data/last_millenium_solar/tsi_WLS.txt'
WSL_solar_with_back = np.genfromtxt(file,skip_header = 4, usecols=(0, 1))
WSL_solar_no_back = np.genfromtxt(file,skip_header = 4, usecols=(0, 2))

#add one more year of data to SBF_solar so it stretched to 1850...
SBF_solar_tmp = SBF_solar_in.copy()
SBF_solar = np.empty([SBF_solar_tmp.shape[0]+1,SBF_solar_tmp.shape[1]])
SBF_solar[0:-1,0] = SBF_solar_tmp[:,0]
SBF_solar[0:-1,1] = SBF_solar_tmp[:,1]
SBF_solar[-1,0] = 1850
SBF_solar[-1,1] = SBF_solar[-2,1]

#add one more year of data to VSK_solar so it stretched to 1850...
VSK_solar_tmp = VSK_solar_in.copy()
VSK_solar = np.empty([VSK_solar_tmp.shape[0]+1,VSK_solar_tmp.shape[1]])
VSK_solar[0:-1,0] = VSK_solar_tmp[:,0]
VSK_solar[0:-1,1] = VSK_solar_tmp[:,1]
VSK_solar[-1,0] = 1850
VSK_solar[-1,1] = VSK_solar[-2,1]

#NOTE using N Hem forcings here
model_forcing={}
model = 'bcc-csm1-1'
model_forcing[model] = {}
model_forcing[model]['volc'] = GRA
model_forcing[model]['solar'] = VSK_solar
model = 'GISS-E2-R'
model_forcing[model] = {}
model_forcing[model]['volc'] = crowley
model_forcing[model]['solar'] = SBF_solar
model = 'IPSL-CM5A-LR'
model_forcing[model] = {}
model_forcing[model]['volc'] = GRA
model_forcing[model]['solar'] = VSK_solar
model = 'MIROC-ESM'
model_forcing[model] = {}
model_forcing[model]['volc'] = crowley
model_forcing[model]['solar'] = DB_solar_with_back
model = 'MPI-ESM-P'
model_forcing[model] = {}
model_forcing[model]['volc'] = crowley
model_forcing[model]['solar'] = VSK_solar
model = 'MRI-CGCM3'
model_forcing[model] = {}
model_forcing[model]['volc'] = GRA
model_forcing[model]['solar'] = DB_solar_with_back
model = 'CCSM4'
model_forcing[model] = {}
model_forcing[model]['volc'] = GRA
model_forcing[model]['solar'] = VSK_solar
model = 'HadCM3'
model_forcing[model] = {}
model_forcing[model]['volc'] = crowley
model_forcing[model]['solar'] = SBF_solar
model = 'CSIRO-Mk3L-1-2'
model_forcing[model] = {}
model_forcing[model]['volc'] = crowley
model_forcing[model]['solar'] = SBF_solar




#west = -24
#east = -5
#south = 65
#north = 81


west = -24
east = -13
south = 65
north = 67


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
	data_and_forcings[model]['ice_area_n_iceland'] = sea_ice_area.data
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
			cube1 = cube1.extract(iris.Constraint(air_pressure = np.max(cube1.coord(dimensions=1).points)))
			cube1 = extract_years(cube1)
			cube1_b = extract_years(iris.load_cube(directory+model+'_va_past1000_r1i1p1_*.nc'))
			cube1_b = cube1_b.extract(iris.Constraint(air_pressure = np.max(cube1_b.coord(dimensions=1).points)))
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
	data_and_forcings[model]['windspeed_n_iceland'] = cube1.data
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
		cube1 = cube1.extract(iris.Constraint(air_pressure = np.max(cube1.coord(dimensions=1).points)))
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
	data_and_forcings[model]['humidity_n_iceland'] = cube1.data
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
	data_and_forcings[model]['net_surface_downward_sw_n_iceland'] = cube1.data
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
	data_and_forcings[model]['net_TOA_downward_sw_n_iceland'] = cube1.data
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
	data_and_forcings[model]['tas_n_iceland'] = cube1.data
		################################
		#   N Iceland salinity  #
		################################
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
		################################
		#   N Iceland air evaporation  #
		################################
	cube1 = iris.load_cube(directory+model+'_evspsbl_past1000_r1i1p1_*.nc')
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
	coord = arctic_sea_ice_area.coord('time')
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
        T = data_and_forcings[model]['tas_n_iceland'] - 273.15
        Rs = data_and_forcings[model]['net_surface_downward_sw_n_iceland'] * day_sec * 1.9e-6
        Ra = data_and_forcings[model]['net_TOA_downward_sw_n_iceland'] * day_sec * 1.9e-6
        RH = data_and_forcings[model]['humidity_n_iceland']
        u = data_and_forcings[model]['windspeed_n_iceland']
        #note that the part of the relationship '(T + 9.5)**(0.5)' breaks down with negative temperatures (because you can't sqrt negatives)
        #so I've gone back to the full calculation
        sea_ice_fraction = data_and_forcings[model]['ice_area_n_iceland']
        data_and_forcings[model]['reconstructed_evap'] = penman(T,Rs,Ra,RH,u) / day_sec
        data_and_forcings[model]['reconstructed_evap_multiplied_by_seaice'] = data_and_forcings[model]['reconstructed_evap'] * (-1.0 * sea_ice_fraction/100.0) 
        T_temp = T.copy()
        T_temp = T_temp * 0.0 +np.mean(T)
        data_and_forcings[model]['reconstructed_evap_T_const'] = penman(T_temp,Rs,Ra,RH,u) / day_sec
        u_temp = u.copy()
        u_temp = u_temp * 0.0 +np.mean(u)
        data_and_forcings[model]['reconstructed_evap_wind_const'] = penman(T,Rs,Ra,RH,u_temp) / day_sec
        RH_temp = RH.copy()
        RH_temp = RH_temp * 0.0 +np.mean(RH)
        data_and_forcings[model]['reconstructed_evap_humidity_const'] = penman(T,Rs,Ra,RH_temp,u) / day_sec
        Rs_temp = Rs.copy()
        Rs_temp = Rs_temp * 0.0 +np.mean(Rs)
        data_and_forcings[model]['reconstructed_evap_surface_radiation_const'] = penman(T,Rs_temp,Ra,RH,u) / day_sec
        Ra_temp = Ra.copy()
        Ra_temp = Ra_temp * 0.0 +np.mean(Ra)
        data_and_forcings[model]['reconstructed_evap_TOA_radiation_const'] = penman(T,Rs,Ra_temp,RH,u) / day_sec
        T_temp = T.copy()
        T_temp = T_temp * 0.0 +np.mean(T)
        Rs_temp = Rs.copy()
        Rs_temp = Rs_temp * 0.0 +np.mean(Rs)
        data_and_forcings[model]['reconstructed_evap_TandRs_const'] = penman(T_temp,Rs_temp,Ra,RH,u) / day_sec


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
        data = scipy.signal.filtfilt(b1, a1, data)
        data = scipy.signal.filtfilt(b2, a2, data)
        data = (data-np.min(data))/(np.max(data)-np.min(data))
        for j,yr in enumerate(tmp_years):
            loc = np.where(data_and_forcings[model]['year'] == yr)
            if np.size(loc) != 0:
                out[i,j] = data[loc]
    return scipy.stats.nanmean(out,axis = 0)

def average_accross_models2(tmp_years,data_and_forcings,models,variable):
    out = np.empty([np.size(models),np.size(tmp_years)]) * 0.0 + np.NAN
    for i,model in enumerate(models):
        data = data_and_forcings[model][variable]
        #data = scipy.signal.filtfilt(b1, a1, data)
        #data = scipy.signal.filtfilt(b2, a2, data)
        #data = (data-np.min(data))/(np.max(data)-np.min(data))
        for j,yr in enumerate(tmp_years):
            loc = np.where(data_and_forcings[model]['year'] == yr)
            if np.size(loc) != 0:
                out[i,j] = data[loc]
    return scipy.stats.nanmean(out,axis = 0)


for model in models:
    plt.plot(data_and_forcings[model]['evap_n_iceland']-np.mean(data_and_forcings[model]['evap_n_iceland']),'b')
    plt.plot(data_and_forcings[model]['reconstructed_evap']-np.mean(data_and_forcings[model]['reconstructed_evap']),'r')
    plt.show()


reconstructed_evap_multiplied_by_seaice_mean = average_accross_modelsb(tmp_years,data_and_forcings,models,'reconstructed_evap_multiplied_by_seaice')

ice_area_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'ice_area_n_iceland')
sw_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'net_surface_downward_sw_n_iceland')
tas_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'tas_n_iceland')
sos_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'sos_n_iceland')
evap_n_iceland_mean = average_accross_models(tmp_years,data_and_forcings,models,'evap_n_iceland')
evap_n_iceland_mean_not_normal = average_accross_models2(tmp_years,data_and_forcings,models,'evap_n_iceland')
reconstructed_evap_mean = average_accross_models2(tmp_years,data_and_forcings,models,'reconstructed_evap')
reconstructed_evap_T_const_mean = average_accross_models2(tmp_years,data_and_forcings,models,'reconstructed_evap_T_const')
reconstructed_evap_TandRs_const_mean = average_accross_models2(tmp_years,data_and_forcings,models,'reconstructed_evap_TandRs_const')
reconstructed_evap_multiplied_by_seaice = average_accross_models2(tmp_years,data_and_forcings,models,'reconstructed_evap_multiplied_by_seaice')
reconstructed_evap_wind_const_mean = average_accross_models2(tmp_years,data_and_forcings,models,'reconstructed_evap_wind_const')
reconstructed_evap_humidity_const_mean = average_accross_models2(tmp_years,data_and_forcings,models,'reconstructed_evap_humidity_const')
reconstructed_evap_surface_radiation_const_mean = average_accross_models2(tmp_years,data_and_forcings,models,'reconstructed_evap_surface_radiation_const')
reconstructed_evap_TOA_radiation_const_mean = average_accross_models2(tmp_years,data_and_forcings,models,'reconstructed_evap_TOA_radiation_const')


#########################################################################################
#                            SPATIAL COMPOSITES                                         #
#########################################################################################




offset = 0
volc_threshold = 0.025
solar_threshold = 0 #because after filtering it varies around 0



for model in models:
	print model
	cube1 = iris.load_cube(directory+model+'_tas_past1000_r1i1p1_*.nc')
	cube1 = extract_years(cube1)
	################################
	#  time  					   #
	################################
	coord = cube1.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	data_and_forcings[model]['year'] = year
        ################################
        #   salinity                   #
        ################################
	#1) does model have surface-level salinity?
	try:
		cube1 = iris.load_cube(directory+model+'_sos_past1000_r1i1p1_*.nc')
		if np.size(np.shape(cube1)) == 4:
			cube1 = cube1[:,0,:,:]
	except:
			#2) does model have 3D salinity (so)?
		cube1 = iris.load_cube(directory+model+'_so_past1000_r1i1p1_*.nc')
		cube1 = cube1.extract(iris.Constraint(str(cube1.coord(dimensions=1).standard_name)+' = '+str(np.min(cube1.coord(dimensions=1).points))))
	cube1b = cube1.copy()
	cube1b.data = scipy.signal.filtfilt(b1, a1, cube1.data,axis = 0)
	cube1b.data = scipy.signal.filtfilt(b2, a2, cube1b.data,axis = 0)
	cube1b.data = np.ma.masked_array(cube1b.data)
	cube1b.data.mask = cube1.data.mask
	cube1b = extract_years(cube1b)
	cube2b = cube1b.intersection(longitude = (west, east))
	cube2b = cube2b.intersection(latitude = (south, north))
	try:
		cube2b.coord('latitude').guess_bounds()
		cube2b.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(cube2b)
	ts_variable = cube2b.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas).data
	ts_variable_sorted = ts_variable.copy()
	ts_variable_sorted.sort()
	threshold = ts_variable_sorted[-200]
	max_loc1 = np.where(ts_variable >= threshold)   
	threshold = ts_variable_sorted[200]
	min_loc1 = np.where(ts_variable <= threshold)  
	west_na = -90
	east_na = 20
	south_na =0
	north_na = 90
	cube1b = cube1b.intersection(longitude = (west_na, east_na))
	cube1b = cube1b.intersection(latitude = (south_na, north_na))
	data_and_forcings[model]['sos_highest_100'] = cube1b[max_loc1]
	data_and_forcings[model]['sos_lowest_100'] = cube1b[min_loc1]
	min_loc = np.where(ts_variable <= np.mean(ts_variable) - np.std(ts_variable))
	max_loc = np.where(ts_variable >= np.mean(ts_variable) + np.std(ts_variable))    
	loc_min2 = np.in1d(year,year[min_loc]+offset)
	loc_max2 = np.in1d(year,year[max_loc]+offset)
	data_and_forcings[model]['sos_sos_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['sos_sos_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
# 	west_na = -90
# 	east_na = 20
# 	south_na =0
# 	north_na = 90
# 	cube1b = cube1b.intersection(longitude = (west_na, east_na))
# 	cube1b = cube1b.intersection(latitude = (south_na, north_na))
# 	data_and_forcings[model]['sos_sos_composite_high_eofs'] = Eof(cube1b[loc_max2])
	################################
	#   tas                   #
	################################
	cube1 = iris.load_cube(directory+model+'_tas_past1000_r1i1p1_*.nc')
	cube1b = cube1.copy()
	cube1b.data = scipy.signal.filtfilt(b, a, cube1.data,axis = 0)
	cube1b.data = scipy.signal.filtfilt(b2, a2,cube1b.data ,axis = 0)
	cube1b = extract_years(cube1b)
	data_and_forcings[model]['sos_tas_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['sos_tas_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
	west_na = -90
	east_na = 20
	south_na =0
	north_na = 90
	cube1b = cube1b.intersection(longitude = (west_na, east_na))
	cube1b = cube1b.intersection(latitude = (south_na, north_na))
	data_and_forcings[model]['tas_highest_100'] = cube1b[max_loc1]
	data_and_forcings[model]['tas_lowest_100'] = cube1b[min_loc1]
	################################
	#   evap                   #
	################################
	cube1 = iris.load_cube(directory+model+'_evspsbl_past1000_r1i1p1_*.nc')
	cube1b = cube1.copy()
	cube1b.data = scipy.signal.filtfilt(b, a, cube1.data,axis = 0)
	cube1b.data = scipy.signal.filtfilt(b2, a2,cube1b.data ,axis = 0)
	cube1b = extract_years(cube1b)
	data_and_forcings[model]['sos_evap_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['sos_evap_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
	west_na = -90
	east_na = 20
	south_na =0
	north_na = 90
	cube1b = cube1b.intersection(longitude = (west_na, east_na))
	cube1b = cube1b.intersection(latitude = (south_na, north_na))
	data_and_forcings[model]['evap_highest_100'] = cube1b[max_loc1]	
	################################
	#   SW                         #
	################################
	cube1 = iris.load_cube(directory+model+'_rsds_past1000_r1i1p1_*.nc')
	cube2 = iris.load_cube(directory+model+'_rsus_past1000_r1i1p1_*.nc')
	cube1 = cube1 - cube2
	cube1b = cube1.copy()
	cube1b.data = scipy.signal.filtfilt(b, a, cube1.data,axis = 0)
	cube1b.data = scipy.signal.filtfilt(b2, a2,cube1b.data ,axis = 0)
	cube1b = extract_years(cube1b)
	data_and_forcings[model]['sos_sw_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['sos_sw_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
	west_na = -90
	east_na = 20
	south_na =0
	north_na = 90
	cube1b = cube1b.intersection(longitude = (west_na, east_na))
	cube1b = cube1b.intersection(latitude = (south_na, north_na))
	data_and_forcings[model]['sw_highest_100'] = cube1b[max_loc1]
	################################
	#  volcanic					   #
	################################
	data_and_forcings[model]['volc'] = data_and_forcings[model]['year'] * 0.0 + np.NAN
	volc_data = model_forcing[model]['volc'][:,1]
	volc_year = model_forcing[model]['volc'][:,0]
	volc_year_floor = np.floor(volc_year)
	volc_year_floor_unique = np.unique(volc_year_floor)
	volc_data2 = np.zeros(np.size(volc_year_floor_unique)) * 0.0 + np.NAN
	for i,yr in enumerate(volc_year_floor_unique):
		loc = np.where(volc_year == yr)
		volc_data2[i] = np.mean(volc_data[loc])
	for i,yr in enumerate(year):
		loc = np.where(volc_year_floor_unique == yr)
		data_and_forcings[model]['volc'][i] = volc_data2[loc]
	################################
	#  solar					   #
	################################
	data_and_forcings[model]['solar'] = data_and_forcings[model]['year'] * 0.0 + np.NAN
	solar_data = model_forcing[model]['solar'][:,1]
	solar_year = model_forcing[model]['solar'][:,0]
	solar_year_floor = np.floor(solar_year)
	solar_year_floor_unique = np.unique(solar_year_floor)
	solar_data2 = np.zeros(np.size(solar_year_floor_unique)) * 0.0 + np.NAN
	for i,yr in enumerate(solar_year_floor_unique):
		loc = np.where(solar_year == yr)
		solar_data2[i] = np.mean(solar_data[loc])
	for i,yr in enumerate(year):
		loc = np.where(solar_year_floor_unique == yr)
		data_and_forcings[model]['solar'][i] = solar_data2[loc]
# 	################################
# 	#   air temperature  #
# 	################################
# 	cube1 = iris.load_cube(directory+model+'_tas_past1000_r1i1p1_*.nc')
# 	cube1 = extract_years(cube1)
# 	if str(cube1.coord(dimensions=1).standard_name) == 'depth':
# 		cube1 = cube1.extract(iris.Constraint(depth = 0))
# 	cube1b = cube1.copy()
# 	cube1b.data = scipy.signal.filtfilt(b, a, cube1.data,axis = 0)
# 	cube1b.data = scipy.signal.filtfilt(b2, a2,cube1b.data ,axis = 0)
# 	cube1b = extract_years(cube1b)
# 	################################
# 	#   airtas eofs                #
# 	################################
#         #data_and_forcings[model]['tas_eof'] = Eof(cube1b)
# 	################################
# 	#   evaporation                #
# 	################################
# 	cube2 = iris.load_cube(directory+model+'_evspsbl_past1000_r1i1p1_*.nc')
# 	cube2 = extract_years(cube2)
# 	cube2b = cube2.copy()
# 	cube2b.data = scipy.signal.filtfilt(b, a, cube2.data,axis = 0)
# 	cube2b.data = scipy.signal.filtfilt(b2, a2,cube2b.data ,axis = 0)
# 	cube2b = extract_years(cube2b)
#         #coord = cube2.coord('time')
# 	#dt = coord.units.num2date(coord.points)
# 	#year2 = np.array([coord.units.num2date(value).year for value in coord.points])
#         #common_year = np.in1d(year,year2)
# 	################################
# 	#       volc composites        #
# 	################################
# 	ts_variable = data_and_forcings[model]['volc']
# 	ts_variable = ts_variable/np.max(ts_variable)
# 	#ts_variable = rm.running_mean(ts_variable,smoothing_val)
# 	min_loc = np.where(ts_variable <= volc_threshold)
# 	max_loc = np.where(ts_variable >= volc_threshold)    
# 	loc_min2 = np.in1d(year,year[min_loc]+offset)
# 	loc_max2 = np.in1d(year,year[max_loc]+offset)
# 	data_and_forcings[model]['tas_volc_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
# 	data_and_forcings[model]['tas_volc_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
# 	data_and_forcings[model]['evap_volc_composite_high'] = cube2b[loc_max2].collapsed('time',iris.analysis.MEAN)
# 	data_and_forcings[model]['evap_volc_composite_low'] = cube2b[loc_min2].collapsed('time',iris.analysis.MEAN)
# 	################################
# 	#       solar composites       #
# 	################################
# 	ts_variable = data_and_forcings[model]['solar']
# 	ts_variable = scipy.signal.filtfilt(b, a, ts_variable,axis = 0)
# 	ts_variable = scipy.signal.filtfilt(b2, a2,ts_variable ,axis = 0)
# 	#ts_variable = rm.running_mean(ts_variable,smoothing_val)
# 	min_loc = np.where(ts_variable <= solar_threshold)
# 	max_loc = np.where(ts_variable >= solar_threshold)    
# 	loc_min2 = np.in1d(year,year[min_loc]+offset)
# 	loc_max2 = np.in1d(year,year[max_loc]+offset)
# 	data_and_forcings[model]['tas_solar_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
# 	data_and_forcings[model]['tas_solar_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
# 	data_and_forcings[model]['evap_solar_composite_high'] = cube2b[loc_max2].collapsed('time',iris.analysis.MEAN)
# 	data_and_forcings[model]['evap_solar_composite_low'] = cube2b[loc_min2].collapsed('time',iris.analysis.MEAN)




def composite_mean(models,data_and_forcings,variable):
	composite_mean_high = data_and_forcings[models[0]][variable].copy()
	composite_mean_data_high = composite_mean_high.data.copy() * 0.0
	i = 0
	for model in models:
			i += 1
			print model
			composite_mean_data_high += data_and_forcings[model][variable].data
	composite_mean_high.data = composite_mean_data_high
	return composite_mean_high / i






sos_sos_composite_mean_high = composite_mean(models,data_and_forcings,'sos_sos_composite_high')
sos_sos_composite_mean_low = composite_mean(models,data_and_forcings,'sos_sos_composite_low')
sos_evap_composite_mean_high = composite_mean(models,data_and_forcings,'sos_evap_composite_high')
sos_tas_composite_mean_high = composite_mean(models,data_and_forcings,'sos_tas_composite_high')
sos_sw_composite_mean_high = composite_mean(models,data_and_forcings,'sos_sw_composite_high')
sos_evap_composite_mean_low = composite_mean(models,data_and_forcings,'sos_evap_composite_low')
sos_tas_composite_mean_low = composite_mean(models,data_and_forcings,'sos_tas_composite_low')
sos_sw_composite_mean_low = composite_mean(models,data_and_forcings,'sos_sw_composite_low')

# volc_composite_mean_high = composite_mean(models,data_and_forcings,'tas_volc_composite_high')
# volc_composite_mean_low = composite_mean(models,data_and_forcings,'tas_volc_composite_low')
# solar_composite_mean_high = composite_mean(models,data_and_forcings,'tas_solar_composite_high')
# solar_composite_mean_low = composite_mean(models,data_and_forcings,'tas_solar_composite_low')

# volc_composite_mean_high_evap = composite_mean(models,data_and_forcings,'evap_volc_composite_high')
# volc_composite_mean_low_evap = composite_mean(models,data_and_forcings,'evap_volc_composite_low')
# solar_composite_mean_high_evap = composite_mean(models,data_and_forcings,'evap_solar_composite_high')
# solar_composite_mean_low_evap = composite_mean(models,data_and_forcings,'evap_solar_composite_low')



'''

###################################################################################################
#                         PLOTTING                                                                #
###################################################################################################



###################################################################################################
#                         Figure 2                                                               #
###################################################################################################


alph = 0.2
wdth = 2

plt.close('all')
fig = plt.figure(figsize=(10,10))

###
#Top Panel
###

ax11 = fig.add_subplot(311)

data = sos_n_iceland_mean
l11a = ax11.plot(tmp_years,data,'r',linewidth = 2,alpha = 0.75,label = 'CMIP5/PMIP3 ensmeble mean GIN Sea salinity')
ax11.set_ylabel('Normalised salinity')

data = evap_n_iceland_mean
ax12 = ax11.twinx()
l12a = ax12.plot(tmp_years,data,'b',linewidth = 2,alpha = 0.75,label = 'CMIP5/PMIP3 ensmeble mean GIN Sea evaporation')
ax12.set_ylabel('Normalised evaporation')

lns = l11a+l12a
labs = [l.get_label() for l in lns]
ax11.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':12})
ax11.set_xlim([950,1850])

###
#2nd Panel
###

ax21 = fig.add_subplot(312)

data = evap_n_iceland_mean_not_normal * day_sec
l21a = ax21.plot(tmp_years,data,'b',linewidth = 2,alpha = 0.75,label = 'CMIP5/PMIP3 ensmeble mean GIN Sea evaporation')
ax21.set_ylabel('Evaporation anomaly (kgm$^{-2}$ day$^{-1}$)')

data = reconstructed_evap_mean * day_sec
l21b = ax21.plot(tmp_years,data,'g',linewidth = 2,alpha = 0.75,label = 'CMIP5/PMIP3 ensmeble mean evaportaion recalculated after Penman equation')

data = reconstructed_evap_T_const_mean * day_sec
l21c = ax21.plot(tmp_years,data,'y',linewidth = 2,alpha = 0.75,label = 'evaportaion recalculated after Penman equation with air temperature constant')


data = reconstructed_evap_TandRs_const_mean * day_sec
l21d = ax21.plot(tmp_years,data,'k',linewidth = 2,alpha = 0.75,label = 'evaportaion recalculated after Penman equation with air temperature and shortwave constant')


lns = l21a + l21b + l21c + l21d
labs = [l.get_label() for l in lns]
ax21.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':12})

ax21.set_xlim([950,1850])

plt.savefig('/home/ph290/Documents/figures/fig2_so_far.png')




plt.close('all')
x = evap_n_iceland_mean_not_normal * day_sec
y = reconstructed_evap_mean * day_sec

x = x - np.mean(x)
y = y - np.mean(y)

xsort = np.argsort(x)
X = x[xsort]
Y = y[xsort]

Xb = sm.add_constant(X)
model = sm.OLS(Y,Xb)
results = model.fit()
r2_1 = results.rsquared

st, data, ss2 = summary_table(results, alpha=0.001)
fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x,y,c=z,s=50,edgecolor = '', cmap='Blues',alpha = 0.5)
l1 = plt.plot(X, fittedvalues, 'b', lw=2,label = 'Evaporation recalculated using a simplified form of the Penman equation\nR$^2$ = '+str(np.round(r2_1*100)/100.0))
plt.plot(X, predict_mean_ci_low, 'k--', lw=2)
plt.plot(X, predict_mean_ci_upp, 'k--', lw=2)

x = evap_n_iceland_mean_not_normal * day_sec
y = reconstructed_evap_T_const_mean * day_sec

x = x - np.mean(x)
y = y - np.mean(y)

xsort = np.argsort(x)
X = x[xsort]
Y = y[xsort]

Xb = sm.add_constant(X)
model = sm.OLS(Y,Xb)
results = model.fit()
r2_2 = results.rsquared

st, data, ss2 = summary_table(results, alpha=0.001)
fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x,y,c=z,s=50,edgecolor = '', cmap='Reds',alpha = 0.5)
l2 = plt.plot(X, fittedvalues, 'R', lw=2,label = 'Recalculated with air temperature kept constant\nR$^2$ = '+str(np.round(r2_2*100)/100.0))
plt.plot(X, predict_mean_ci_low, 'k--', lw=2)
plt.plot(X, predict_mean_ci_upp, 'k--', lw=2)

x = evap_n_iceland_mean_not_normal * day_sec
y = reconstructed_evap_TandRs_const_mean * day_sec

x = x - np.mean(x)
y = y - np.mean(y)

xsort = np.argsort(x)
X = x[xsort]
Y = y[xsort]

Xb = sm.add_constant(X)
model = sm.OLS(Y,Xb)
results = model.fit()
r2_3 = results.rsquared

st, data, ss2 = summary_table(results, alpha=0.001)
fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x,y,c=z,s=50,edgecolor = '', cmap='Greens',alpha = 0.5)
l3 = plt.plot(X, fittedvalues, 'g', lw=2,label = 'Recalculated with air temperature and shortwave kept constant\nR$^2$ = '+str(np.round(r2_3*100)/100.0))
plt.plot(X, predict_mean_ci_low, 'k--', lw=2)
plt.plot(X, predict_mean_ci_upp, 'k--', lw=2)

#range = 0.07
#plt.xlim([-1*range,range])
#plt.ylim([-1*range,range])
plt.xlabel('Evaporation anomaly (kg m$^{-2}$ day$^{-1}$)')
plt.ylabel('Evaporation anomaly recalculated\nusing the Penman equation (kg m$^{-2}$ day$^{-1}$)')

lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'upper left', fancybox=True, framealpha=0.1,prop={'size':9})

plt.plot(np.arange(-10,10),np.arange(-10,10),'k',linewidth = 4)

plt.tight_layout()

# plt.show()
plt.savefig('/home/ph290/Documents/figures/palaeoamo/figure_2b.png')



###################################################################################################
#                         Figure 3                                                                #
###################################################################################################



for model in models:
	print model
	################################
	#   air temperature  #
	################################
	cube1 = iris.load_cube(directory+model+'_tas_past1000_r1i1p1_*.nc')
	cube1b = cube1.copy()
	cube1b.data = scipy.signal.filtfilt(b1, a1, cube1.data,axis = 0)
	cube1b = extract_years(cube1b)
	################################
	#   SW                         #
	################################
	cube1 = iris.load_cube(directory+model+'_rsds_past1000_r1i1p1_*.nc')
	cube2 = iris.load_cube(directory+model+'_rsus_past1000_r1i1p1_*.nc')
	cube1 = cube1 - cube2
	cube2b = cube1.copy()
	cube2b.data = scipy.signal.filtfilt(b1, a1, cube1.data,axis = 0)
	cube2b = extract_years(cube2b)
	################################
	#  time  					   #
	################################
	coord = cube1b.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	data_and_forcings[model]['year'] = year
	################################
	#       volc composites        #
	################################
	ts_variable = data_and_forcings[model]['volc']
	ts_variable = ts_variable/np.max(ts_variable)
	#ts_variable = rm.running_mean(ts_variable,smoothing_val)
	max_loc = np.where(ts_variable >= volc_threshold)
	min_loc = np.where(ts_variable >= 0.00)       
	loc_max2 = np.in1d(year,year[max_loc]+offset)
	loc_min2 = np.in1d(year,year[min_loc]+offset)
	data_and_forcings[model]['tas_volc_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
	loc_min2 = np.in1d(year,year[min_loc]+offset)
	data_and_forcings[model]['tas_volc_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['sw_volc_composite_high'] = cube2b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['sw_volc_composite_low'] = cube2b[loc_min2].collapsed('time',iris.analysis.MEAN)


for model in models:
	print model
	################################
	#   air temperature  #
	################################
	cube1 = iris.load_cube(directory+model+'_tas_past1000_r1i1p1_*.nc')
	cube1 = extract_years(cube1)
	cube1b = cube1.copy()
	cube1b.data = scipy.signal.filtfilt(b1, a1, cube1.data,axis = 0)
	cube1b = extract_years(cube1b)
	################################
	#   SW                         #
	################################
	cube1 = iris.load_cube(directory+model+'_rsds_past1000_r1i1p1_*.nc')
	cube2 = iris.load_cube(directory+model+'_rsus_past1000_r1i1p1_*.nc')
	cube1 = cube1 - cube2
	cube2b = cube1.copy()
	cube2b.data = scipy.signal.filtfilt(b1, a1, cube1.data,axis = 0)
	cube2b = extract_years(cube2b)
	################################
	#  time  					   #
	################################
	coord = cube1b.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	data_and_forcings[model]['year'] = year
	################################
	#       solar composites        #
	################################
	ts_variable = data_and_forcings[model]['solar']
	ts_variable = scipy.signal.filtfilt(b1, a1, ts_variable,axis = 0)
	ts_variable = scipy.signal.filtfilt(b3, a3, ts_variable,axis = 0)
	ts_variable = ts_variable+np.abs(np.min(ts_variable))
	ts_variable = ts_variable/np.max(ts_variable)
	max_loc = np.where(ts_variable >= 0.66)    
	min_loc = np.where(ts_variable >= 0.33) 
	loc_max2 = np.in1d(year,year[max_loc]+offset)
	data_and_forcings[model]['tas_solar_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
	loc_min2 = np.in1d(year,year[min_loc]+offset)
	data_and_forcings[model]['tas_solar_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['sw_solar_composite_high'] = cube2b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['sw_solar_composite_low'] = cube2b[loc_min2].collapsed('time',iris.analysis.MEAN)



volc_composite_mean_high = composite_mean(models,data_and_forcings,'tas_volc_composite_high')
volc_composite_mean_low = composite_mean(models,data_and_forcings,'tas_volc_composite_low')
solar_composite_mean_high = composite_mean(models,data_and_forcings,'tas_solar_composite_high')
solar_composite_mean_low = composite_mean(models,data_and_forcings,'tas_solar_composite_low')

volc_composite_mean_high_sw = composite_mean(models,data_and_forcings,'sw_volc_composite_high')
volc_composite_mean_low_sw = composite_mean(models,data_and_forcings,'sw_volc_composite_low')
solar_composite_mean_high_sw = composite_mean(models,data_and_forcings,'sw_solar_composite_high')
solar_composite_mean_low_sw = composite_mean(models,data_and_forcings,'sw_solar_composite_low')

b4, a4 = butter_highpass(1.0/6, 1.0,2)

meaned_solar = np.zeros([np.size(models),1001]) * 0.0 + np.NAN

for i,model in enumerate(models):
	ts_variable = data_and_forcings[model]['solar']
	ts_variable = ts_variable+np.min(ts_variable)
	ts_variable = ts_variable/np.max(ts_variable)
	ts_variable = scipy.signal.filtfilt(b1, a1, ts_variable,axis = 0)
	ts_variable = scipy.signal.filtfilt(b4, a4, ts_variable,axis = 0)
	meaned_solar[i,0:np.size(ts_variable)] = ts_variable


meaned_solar2 = scipy.stats.nanmean(meaned_solar,axis=0)
meaned_solar2 = meaned_solar2+np.abs(np.min(meaned_solar2))
meaned_solar2 = meaned_solar2/np.max(meaned_solar2)


meaned_volc = np.zeros([np.size(models),1001]) * 0.0 + np.NAN

for i,model in enumerate(models):
	ts_variable = data_and_forcings[model]['volc']
	ts_variable = ts_variable/np.max(ts_variable)
	meaned_volc[i,0:np.size(ts_variable)] = ts_variable



meaned_volc2 = scipy.stats.nanmean(meaned_volc,axis=0)
meaned_volc2 = meaned_volc2+np.abs(np.min(meaned_volc2))
meaned_volc2 = meaned_volc2/np.max(meaned_volc2)



###
# tas
###

plt.close('all')
fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

l1 = ax1.plot(np.arange(850,1851),tas_n_iceland_mean,'b',linewidth = 2,alpha = 0.75,label = 'GIN Sea air temperature')
#ax1b = ax1.twinx()
x = meaned_solar2*0.2 - rmp.running_mean_post(meaned_volc2,7)
x = scipy.signal.filtfilt(b1, a1, x,axis = 0)+0.55
x2 = meaned_solar2*0.2
x2 = scipy.signal.filtfilt(b1, a1, x2,axis = 0)+0.55
#ax1b.plot(np.arange(850,1851),x,'r',linewidth = 2,alpha = 0.75)
#ax1b.plot(np.arange(850,1851),x2,'g',linewidth = 2,alpha = 0.75)
l2 = ax1.plot(np.arange(850,1851),x,'r',linewidth = 2,alpha = 0.75,label = 'Simple model of air temperature using volcanic and solar index')
l3 = ax1.plot(np.arange(850,1851),x2,'g',linewidth = 2,alpha = 0.75,label = 'Simple model of air temperature using just solar index')
ax1.set_xlabel('calendar year')
ax1.set_ylabel('temperature ($^{\circ}$C)')
ax1.set_xlim([950,1850])

lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':12})

brewer_cmap = 'bwr'
levels = np.linspace(-0.2,0.2,31)

ax2 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, projection = ccrs.PlateCarree())
ax3 = plt.subplot2grid((3, 2), (0, 1), rowspan=2, projection = ccrs.PlateCarree())

to_plot = volc_composite_mean_high-volc_composite_mean_low
lons = to_plot.coord('longitude').points
lats = to_plot.coord('latitude').points
# ax1 = plt.subplot(121, projection = ccrs.PlateCarree())
ax2.set_extent((-90.0, 20.0, 0.0, 90.0), crs=ccrs.PlateCarree())
contour = ax2.contourf(lons, lats, to_plot.data,levels=levels,cmap=brewer_cmap)
cartopy.feature.LAND.scale='50m'
ax2.add_feature(cartopy.feature.LAND)
ax2.coastlines(resolution='50m')
ax2.set_title('Composite of high minus\nlow volcanic years')

to_plot = solar_composite_mean_low - solar_composite_mean_high
lons = to_plot.coord('longitude').points
lats = to_plot.coord('latitude').points
# ax2 = plt.subplot(122, projection = ccrs.PlateCarree())
ax3.set_extent((-90.0, 20.0, 0.0, 90.0), crs=ccrs.PlateCarree())
contour = ax3.contourf(lons, lats, to_plot.data,levels=levels,cmap=brewer_cmap)
cartopy.feature.LAND.scale='50m'
ax3.add_feature(cartopy.feature.LAND)
ax3.coastlines(resolution='50m')
ax3.set_title('Composite of high minus\nlow solar years')


cax = fig.add_axes([0.1, 0.45, 0.85, 0.03])

cbar = plt.colorbar(contour, cax=cax, ticks=[-0.2, -0.1,0, 0.1,0.2],orientation = 'horizontal')
cbar.set_label('temperature anomaly ($^{\circ}$C)')

plt.tight_layout()


# plt.show()
plt.savefig('/home/ph290/Documents/figures/palaeoamo/figure_3.png')


#########################################################

###
# sw
###

b5b, a5b = butter_highpass(1.0/6, 1.0,2)
b5a, a5a = butter_lowpass(1.0/100, 1.0,2)
b6, a6 = butter_lowpass(1.0/50, 1.0,2)


meaned_solar = np.zeros([np.size(models),1001]) * 0.0 + np.NAN

for i,model in enumerate(models):
	ts_variable = data_and_forcings[model]['solar']
	ts_variable = ts_variable+np.min(ts_variable)
	ts_variable = ts_variable/np.max(ts_variable)
	#ts_variable = scipy.signal.filtfilt(b5a, a5a, ts_variable,axis = 0)
	ts_variable = scipy.signal.filtfilt(b5b, a5b, ts_variable,axis = 0)
	meaned_solar[i,0:np.size(ts_variable)] = ts_variable


meaned_solar3 = scipy.stats.nanmean(meaned_solar,axis=0)
meaned_solar3 = meaned_solar3+np.abs(np.min(meaned_solar3))
meaned_solar3 = meaned_solar3/np.max(meaned_solar3)


plt.close('all')
fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

l1 = ax1.plot(np.arange(850,1851),sw_n_iceland_mean,'b',linewidth = 2,alpha = 0.75,label = 'GIN Sea shortwave radiation')
#ax1b = ax1.twinx()
x = meaned_solar3*1000 - rmp.running_mean_post(meaned_volc2,7)
x = scipy.signal.filtfilt(b6, a6, x,axis = 0)+0.5
x2 = meaned_solar3*1000
x2 = scipy.signal.filtfilt(b1, a1, x2,axis = 0)+0.5
x = ice_area_n_iceland_mean*(-1.0)+1.0
#ax1b.plot(np.arange(850,1851),x,'r',linewidth = 2,alpha = 0.75)
#ax1b.plot(np.arange(850,1851),x2,'g',linewidth = 2,alpha = 0.75)
l2 = ax1.plot(np.arange(850,1851),x,'r',linewidth = 2,alpha = 0.75,label = 'Simple model of air temperature using volcanic and solar index')
l3 = ax1.plot(np.arange(850,1851),x2,'g',linewidth = 2,alpha = 0.75,label = 'Simple model of air temperature using just solar index')
ax1.set_xlabel('calendar year')
ax1.set_ylabel('shortwave radiation (W m$^{-2}$)')
ax1.set_xlim([950,1850])
plt.show(block = False)


lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':12})


brewer_cmap = 'bwr'
levels = np.linspace(-1.0,1.0,31)

ax2 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, projection = ccrs.PlateCarree())
ax3 = plt.subplot2grid((3, 2), (0, 1), rowspan=2, projection = ccrs.PlateCarree())

to_plot = volc_composite_mean_high_sw-volc_composite_mean_low_sw
lons = to_plot.coord('longitude').points
lats = to_plot.coord('latitude').points
# ax1 = plt.subplot(121, projection = ccrs.PlateCarree())
ax2.set_extent((-90.0, 20.0, 0.0, 90.0), crs=ccrs.PlateCarree())
contour = ax2.contourf(lons, lats, to_plot.data,levels=levels,cmap=brewer_cmap)
cartopy.feature.LAND.scale='50m'
ax2.add_feature(cartopy.feature.LAND)
ax2.coastlines(resolution='50m')
ax2.set_title('Composite of high minus\nlow volcanic years')

to_plot = solar_composite_mean_low_sw - solar_composite_mean_high_sw
lons = to_plot.coord('longitude').points
lats = to_plot.coord('latitude').points
# ax2 = plt.subplot(122, projection = ccrs.PlateCarree())
ax3.set_extent((-90.0, 20.0, 0.0, 90.0), crs=ccrs.PlateCarree())
contour = ax3.contourf(lons, lats, to_plot.data,levels=levels,cmap=brewer_cmap)
cartopy.feature.LAND.scale='50m'
ax3.add_feature(cartopy.feature.LAND)
ax3.coastlines(resolution='50m')
ax3.set_title('Composite of high minus\nlow solar years')


cax = fig.add_axes([0.1, 0.45, 0.85, 0.03])

cbar = plt.colorbar(contour, cax=cax, ticks=[-0.2, -0.1,0, 0.1,0.2],orientation = 'horizontal')
cbar.set_label('shortwave radiation anomaly (W m$^{-2}$)')

plt.tight_layout()


plt.show()
#plt.savefig('/home/ph290/Documents/figures/palaeoamo/figure_3b.png')












# #Add the model-specific volcanic solar and forcing data to the dictionary
# for model in models:
# 	year = data_and_forcings[model]['year']
# 	data_and_forcings[model]['volc'] = data_and_forcings[model]['year'] * 0.0 + np.NAN
# 	volc_data = model_forcing[model]['volc'][:,1]
# 	volc_year = model_forcing[model]['volc'][:,0]
# 	volc_year_floor = np.floor(volc_year)
# 	volc_year_floor_unique = np.unique(volc_year_floor)
# 	volc_data2 = np.zeros(np.size(volc_year_floor_unique)) * 0.0 + np.NAN
# 	for i,yr in enumerate(volc_year_floor_unique):
# 		loc = np.where(volc_year == yr)
# 		volc_data2[i] = np.mean(volc_data[loc])
# 	for i,yr in enumerate(year):
# 		loc = np.where(volc_year_floor_unique == yr)
# 		data_and_forcings[model]['volc'][i] = volc_data2[loc]
# 	data_and_forcings[model]['solar'] = data_and_forcings[model]['year'] * 0.0 + np.NAN
# 	solar_data = model_forcing[model]['solar'][:,1]
# 	solar_year = model_forcing[model]['solar'][:,0]
# 	solar_year_floor = np.floor(solar_year)
# 	solar_year_floor_unique = np.unique(solar_year_floor)
# 	solar_data2 = np.zeros(np.size(solar_year_floor_unique)) * 0.0 + np.NAN
# 	for i,yr in enumerate(solar_year_floor_unique):
# 		loc = np.where(solar_year == yr)
# 		solar_data2[i] = np.mean(solar_data[loc])
# 	for i,yr in enumerate(year):
# 		loc = np.where(solar_year_floor_unique == yr)
# 		data_and_forcings[model]['solar'][i] = solar_data2[loc]


# r_data_file = '/home/ph290/data0/reynolds/ultra_data.csv'
# r_data = np.genfromtxt(r_data_file,skip_header = 1,delimiter = ',')
# tmp = r_data[:,1]
# tmp = scipy.signal.filtfilt(b, a, tmp)
# tmp = rm.running_mean(tmp,smoothing_val)
# loc = np.where((np.logical_not(np.isnan(tmp))) & (r_data[:,0] >= start_date) & (r_data[:,0] <= end_date))
# tmp = tmp[loc]
# tmp_yr = r_data[loc[0],0]



# model = models[0]
# ice_area_n_iceland_all = np.empty([np.size(data_and_forcings[model]['year']),np.size(models)])
# forcing_all = np.empty([np.size(data_and_forcings[model]['year']),np.size(models)])

# smoothing_val = 20

# for i,model in enumerate(models):
#     plt.close('all')
#     fig, ax1 = plt.subplots()
#     x = data_and_forcings[model]['year']
#     y1 = data_and_forcings[model]['solar']
#     y1 = scipy.signal.filtfilt(b, a, y1)
#     y1 = rm.running_mean(y1,smoothing_val)
#     y2 = data_and_forcings[model]['ice_area_n_iceland']
#     y2 = scipy.signal.filtfilt(b, a, y2)
#     y2 = rm.running_mean(y2,smoothing_val)
#     y3 = data_and_forcings[model]['volc']
#     y3=(y3-np.nanmin(y3))/(np.nanmax(y3)-np.nanmin(y3))
#     #y3 = rmp.running_mean_post(y3,20)
#     ################################
#     y3 = rm.running_mean(y3,20)
#     y3 = np.log(y3+1)
#     y3=(y3-np.nanmin(y3))/(np.nanmax(y3)-np.nanmin(y3))
#     #y3 = scipy.signal.filtfilt(b, a, y3)
#     #y3 = np.log(y3+1)
#     #y3=(y3-np.nanmin(y3))/(np.nanmax(y3)-np.nanmin(y3))
#     plotting_y = (y1*-1.0)+y3*1.5
#     ax1.plot(x,plotting_y,'g')
#     #ax1.plot(x,(y1*1.0),'g.')
#     #ax1.plot(x,y3*2.0,'g--')
#     ax2 = ax1.twinx()
#     ax2.plot(x,y2,'b')
#     ################################
#     #ax3 = ax2.twinx()
#     #ax3.plot(x,y3,'r')
#     # y3b = np.log(y3+1)
#     # y3b[np.where(np.logical_not(np.isfinite(y3b)))] = 0
#     # y1[np.where(np.logical_not(np.isfinite(y1)))] = 0
#     # y = np.column_stack((y1,y3b))
#     # y = sm.add_constant(y)
#     # y2[np.where(np.logical_not(np.isfinite(y2)))] = 0
#     # mlr_model = sm.OLS(y2,y)
#     # results = mlr_model.fit()
#     # ax1.plot(x,y2)
#     # ax1.plot(x,results.params[2]*y3b+results.params[1]*y1+results.params[0])
#     #ax1.scatter((y1*-1.0)+y3,y2)
#     plt.show(block = False)
#     ice_area_n_iceland_all[0:np.size(y2),i] = y2
#     forcing_all[0:np.size(plotting_y),i] = plotting_y


# ice_area_n_iceland_mean = scipy.stats.nanmean(ice_area_n_iceland_all,axis = 1)
# forcing_mean = scipy.stats.nanmean(forcing_all,axis = 1)

# plt.close('all')
# fig, ax1 = plt.subplots()
# ax1.plot(range(850,1851),forcing_mean,'w')
# ax2 = ax1.twinx()
# ax2.plot(range(850,1851),ice_area_n_iceland_mean,'b')
# #ax3 = ax2.twinx()
# #ax3.plot(tmp_yr,tmp,'g')
# plt.show()


