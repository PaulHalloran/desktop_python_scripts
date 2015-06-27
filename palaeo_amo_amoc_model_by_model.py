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




###
#Filter
###


def butter_bandpass(lowcut, fs, order=5):
    nyq = fs
    low = lowcut/nyq
    b, a = scipy.signal.butter(order, low , btype='high',analog = False)
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

###########################################################################################################
#                                                                                                         #
#   start of non stream function bit (stream function stuff done in palaeo_amo_amoc_paper_figures_iv.py)  #
#                                                                                                         #
###########################################################################################################

end_date = end_year = 1850
start_date = start_year = 850
expected_years = np.arange(850,1850)
smoothing_val = 20

with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_IIII.pickle') as f:    models,max_strm_fun,max_strm_fun_26,max_strm_fun_45,model_years,mask1,files,b,a,input_file,resolution,start_date,end_date,location = pickle.load(f)

b, a = butter_bandpass(1.0/100.0, 1.0,2)

print '- NOTE! FGOALS MODEL LEVELS ARE UPSIDE DOWN - DOES THIS MATTER?, check mask figures to explain'

print 'check masks in /home/ph290/Documents/figures'


###
#read in temperature
###

amo_box_tas = []
model_years_tas = []



for i,model in enumerate(models):
	print 'processing: '+model
	file = glob.glob('/media/usb_external1/cmip5/last1000/'+model+'_tos_past1000_r1i1p1_*.nc')
	cube = iris.load_cube(file)
	lon_west = -75
	lon_east = -7.5
	lat_south = 0
	lat_north = 60.0
	cube = cube.intersection(longitude=(lon_west, lon_east))
	cube = cube.intersection(latitude=(lat_south, lat_north))
	cube.coord('latitude').guess_bounds()
	cube.coord('longitude').guess_bounds()
	grid_areas = iris.analysis.cartography.area_weights(cube)
	ts = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
	amo_box_tas.append(ts)
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	years = np.array([coord.units.num2date(value).year for value in coord.points])
	model_years_tas.append(years)



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

#NOTE using global forcings here
model_forcing={}
model = 'bcc-csm1-1'
model_forcing[model] = {}
model_forcing[model]['volc'] = GRA
model_forcing[model]['solar'] = VSK_solar
model = 'GISS-E2-R'
model_forcing[model] = {}
model_forcing[model]['volc'] = 'DEPENDS ON THE ENSEMBLE'
model_forcing[model]['solar'] = 'DEPENDS ON THE ENSEMBLE'
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


###
#Construct dictionaries containing the models to use and the associated stream function and tas. Note just taking the 1st ensmeble from GISS, which using on of the volc forcings etc. (other 'ensemble' members use different forcings etc.)
###

pmip3_tas = {}
pmip3_str = {}
pmip3_year_tas = {}
pmip3_year_str = {}

giss_test = 0

for i,model in enumerate(models):
	if model == 'GISS-E2-R':
		if giss_test == 0:
			pmip3_tas[model] = amo_box_tas[i].data
			pmip3_str[model] = max_strm_fun_26[i]
			pmip3_year_str[model] = model_years[i]
			pmip3_year_tas[model] = model_years_tas[i]
			giss_test += 1
	if model <> 'GISS-E2-R':
		pmip3_tas[model] = amo_box_tas[i].data
		pmip3_str[model] = max_strm_fun_26[i]
		pmip3_year_str[model] = model_years[i]
		pmip3_year_tas[model] = model_years_tas[i]
	
	

#for i,model in enumerate(models):
#	pmip3_tas[model] = amo_box_tas[i].data
#	pmip3_str[model] = max_strmfun_26[i]
#	pmip3_year_str[model] = model_years[i]
#	pmip3_year_tas[model] = model_years_tas[i]
	
#all_models = np.unique(models+models_unique)



print 'have you finished processing the input files for the two following models. Currently downloading extra data'

#hfjohdjklasdjkl
modles = list(models)
models.remove('FGOALS-gl')
#REMOVE FGOALS-gl BECAUSE STARTS IN yr 1000 not 850
models.remove('FGOALS-s2')
models.remove('GISS-E2-R')
#NOTE! FGOALS MODEL LEVELS ARE UPSIDE DOWN - DOES THIS MATTER?, check mask figures to explain'
models.remove('bcc-csm1-1')
models = np.array(models)

all_models = models
#added = does this cause problems?

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_II.pickle', 'w') as f:
#    pickle.dump([all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,all_models], f)

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_II.pickle') as f:
#ll_models_tas_sos_pr    all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,all_models = pickle.load(f)

all_years = np.linspace(850,1850,(1851-850))


directory = '/media/usb_external1/cmip5/last1000/'
#NOTE so_Omon_bcc-csm1-1_past1000_r1i1p1_119001-119912.nc is missing and not on the CMIP5 archive. This causes problems

west = -24
east = -13
south = 65
north = 67



west1 =  -180
east1 = 180
south1 = 80
north1 = 100

west2 = -25
east2 = -15
south2 = 50
north2 = 60

density_data = {}
#models_tas_sos_pr
for model in models:
	print model
# 	try:
	#pressure
	psl_cube = iris.load_cube(directory+model+'_psl_past1000_r1i1p1*.nc')
	psl_cube = extract_years(psl_cube)
	try:
		psl_depths = psl_cube.coord('depth').points
		psl_cube = psl_cube.extract(iris.Constraint(depth = np.min(psl_depths)))
	except:
		print 'no depth coordinate'
	temporary_cube = psl_cube.intersection(longitude = (west1, east1))
	psl_cube_arctic = temporary_cube.intersection(latitude = (south1, north1))
	try:
		psl_cube_arctic.coord('latitude').guess_bounds()
		psl_cube_arctic.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(psl_cube_arctic)
	psl_cube_arctic_mean = psl_cube_arctic.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
        ##
	temporary_cube = psl_cube.intersection(longitude = (west2, east2))
	psl_cube_spg = temporary_cube.intersection(latitude = (south2, north2))
	try:
		psl_cube_spg.coord('latitude').guess_bounds()
		psl_cube_spg.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(psl_cube_spg)
	psl_cube_spg_mean = psl_cube_spg.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
        psl_diff = psl_cube_arctic_mean - psl_cube_spg_mean
	#evap
	evap_cube = iris.load_cube(directory+model+'_evspsbl_past1000_r1i1p1*.nc')
	evap_cube = extract_years(evap_cube)
	try:
		evap_depths = evap_cube.coord('depth').points
		evap_cube = evap_cube.extract(iris.Constraint(depth = np.min(evap_depths)))
	except:
		print 'no evap depth coordinate'
	temporary_cube = evap_cube.intersection(longitude = (west, east))
	evap_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
	try:
		evap_cube_n_iceland.coord('latitude').guess_bounds()
		evap_cube_n_iceland.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(evap_cube_n_iceland)
	evap_cube_n_iceland_mean = evap_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
	#temperature
	t_cube = iris.load_cube(directory+model+'_tos_past1000_r1i1p1*.nc')
	t_cube = extract_years(t_cube)
	try:
		t_depths = t_cube.coord('depth').points
		t_cube = t_cube.extract(iris.Constraint(depth = np.min(t_depths)))
	except:
		print 'no temperature depth coordinate'
	temporary_cube = t_cube.intersection(longitude = (west, east))
	t_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
	try:
		t_cube_n_iceland.coord('latitude').guess_bounds()
		t_cube_n_iceland.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(t_cube_n_iceland)
	t_cube_n_iceland_mean = t_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
	#salinity
	s_cube = iris.load_cube(directory+model+'_sos_past1000_r1i1p1*.nc')
	s_cube = extract_years(s_cube)
	try:
		s_depths = s_cube.coord('depth').points
		s_cube = s_cube.extract(iris.Constraint(depth = np.min(s_depths)))
	except:
		print 'no salinity depth coordinate'
	temporary_cube = s_cube.intersection(longitude = (west, east))
	s_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
	try:
		s_cube_n_iceland.coord('latitude').guess_bounds()
		s_cube_n_iceland.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(s_cube_n_iceland)
	s_cube_n_iceland_mean = s_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
	#precipitation
	pr_cube = iris.load_cube(directory+model+'_pr_past1000_r1i1p1*_regridded.nc')
	pr_cube = extract_years(pr_cube)
	try:
		pr_depths = pr_cube.coord('depth').points
		pr_cube = pr_cube.extract(iris.Constraint(depth = np.min(pr_depths)))
	except:
		print 'no precipitation depth coordinate'
	temporary_cube = pr_cube.intersection(longitude = (west, east))
	pr_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
	try:
		pr_cube_n_iceland.coord('latitude').guess_bounds()
		pr_cube_n_iceland.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(pr_cube_n_iceland)
	pr_cube_n_iceland_mean = pr_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
	#density
	tmp_density = seawater.dens(s_cube_n_iceland_mean,t_cube_n_iceland_mean-273.15)
	test = np.size(t_cube_n_iceland_mean)
	if test <= 1000:
		temp_t_mean = np.mean(t_cube_n_iceland_mean)
	if test > 1000:
		temp_t_mean = np.mean(t_cube_n_iceland_mean[0:1000])
	test = np.size(s_cube_n_iceland_mean)
	if test <= 1000:
		temp_s_mean = np.mean(s_cube_n_iceland_mean)
	if test > 1000:
		temp_s_mean = np.mean(s_cube_n_iceland_mean[0:1000])
	tmp_temp_mean_density = seawater.dens(s_cube_n_iceland_mean,t_cube_n_iceland_mean*0.0+temp_t_mean-273.15)
	tmp_sal_mean_density = seawater.dens(s_cube_n_iceland_mean*0.0+temp_s_mean, t_cube_n_iceland_mean-273.15)
	#years
	coord = t_cube_n_iceland.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	#output
	density_data[model] = {}
	density_data[model]['evspsbl'] = evap_cube_n_iceland_mean
	density_data[model]['temperature'] = t_cube_n_iceland_mean
	density_data[model]['salinity'] = s_cube_n_iceland_mean
	density_data[model]['density'] = tmp_density
	density_data[model]['temperature_meaned_density'] = tmp_temp_mean_density
	density_data[model]['salinity_meaned_density'] = tmp_sal_mean_density
	density_data[model]['precipitation'] = pr_cube_n_iceland_mean
	density_data[model]['years'] = year_tmp
	density_data[model]['psl_arctic'] = psl_cube_arctic_mean
        density_data[model]['psl_spg'] = psl_cube_spg_mean
        density_data[model]['psl_diff'] = psl_diff
# 	except:
# 			print 'model can not be read in'


###
#Averaging the density data
###
	
min_yr = 10000
max_yr = 0
for model in density_data.viewkeys():
	tmp_min_yr = np.min(density_data[model]['years'])
	if 	tmp_min_yr < min_yr:
		min_yr = tmp_min_yr
	tmp_max_yr = np.max(density_data[model]['years'])
	if 	tmp_max_yr > max_yr:
		max_yr = tmp_max_yr
		

mean_density = np.zeros([max_yr+1 - min_yr,len(density_data)])
mean_density[::] = np.NAN
mean_temperature = mean_density.copy()
mean_salinity = mean_density.copy()
temperature_meaned_density = mean_density.copy()
salinity_meaned_density = mean_density.copy()
precipitation = mean_density.copy()
psl_arctic = mean_density.copy()
psl_spg = mean_density.copy()
psl_diff = mean_density.copy()
evap = mean_density.copy()

years = range(min_yr,max_yr+1)
for i,model in enumerate(density_data.viewkeys()):
	tmp_yrs = density_data[model]['years']
	data1 = density_data[model]['density']
	data1 = scipy.signal.filtfilt(b, a, data1)
	data1 = rm.running_mean(data1,smoothing_val)
	x = data1
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data1 = x
	data2 = density_data[model]['temperature']
	data2 = scipy.signal.filtfilt(b, a, data2)
	data2 = rm.running_mean(data2,smoothing_val)
	x = data2
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data2 = x
	data3 = density_data[model]['salinity']
	data3 = scipy.signal.filtfilt(b, a, data3)
	data3 = rm.running_mean(data3,smoothing_val)
	x = data3
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data3 = x
	data4 = density_data[model]['temperature_meaned_density']
	data4 = scipy.signal.filtfilt(b, a, data4)
	data4 = rm.running_mean(data4,smoothing_val)
	x = data4
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data4 = x
	data5 = density_data[model]['salinity_meaned_density']
	data5 = scipy.signal.filtfilt(b, a, data5)
	data5 = rm.running_mean(data5,smoothing_val)
	x = data5
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data5 = x
	data6 = density_data[model]['precipitation']
	data6 = scipy.signal.filtfilt(b, a, data6)
	data6 = rm.running_mean(data6,smoothing_val)
	x = data6
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data6 = x
	data7 = density_data[model]['psl_arctic']
	data7 = scipy.signal.filtfilt(b, a, data7)
	data7 = rm.running_mean(data7,smoothing_val)
	x = data7
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data7 = x
	data8 = density_data[model]['psl_spg']
	data8 = scipy.signal.filtfilt(b, a, data8)
	data8 = rm.running_mean(data8,smoothing_val)
	x = data8
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data8 = x
	data9 = density_data[model]['psl_diff']
	data9 = scipy.signal.filtfilt(b, a, data9)
	data9 = rm.running_mean(data9,smoothing_val)
	x = data9
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data9 = x
	data10 = density_data[model]['evspsbl']
	data10 = scipy.signal.filtfilt(b, a, data10)
	data10 = rm.running_mean(data10,smoothing_val)
	x = data10
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data10 = x
	#assigning this data to 2-D (model/time) array for meaning
	for j,tmp_yr in enumerate(tmp_yrs):
		loc = np.where(tmp_yr == years)
		mean_density[loc,i] = data1[j]
		mean_temperature[loc,i] = data2[j]
		mean_salinity[loc,i] = data3[j]
		temperature_meaned_density[loc,i] = data4[j]
		salinity_meaned_density[loc,i] = data5[j]
		precipitation[loc,i] = data6[j]
                psl_arctic[loc,i] = data7[j]
                psl_spg[loc,i] = data8[j]
                psl_diff[loc,i] = data9[j]
                evap[loc,i] = data10[j]




mean_density = np.ma.masked_invalid(mean_density)
mean_density2 = np.ma.mean(mean_density,axis = 1)
mean_temperature = np.ma.masked_invalid(mean_temperature)
mean_temperature2 = np.ma.mean(mean_temperature,axis = 1)
mean_salinity = np.ma.masked_invalid(mean_salinity)
mean_salinity2 = np.ma.mean(mean_salinity,axis = 1)
temperature_meaned_density = np.ma.masked_invalid(temperature_meaned_density)
temperature_meaned_density2 = np.ma.mean(temperature_meaned_density,axis = 1)
salinity_meaned_density = np.ma.masked_invalid(salinity_meaned_density)
salinity_meaned_density2 = np.ma.mean(salinity_meaned_density,axis = 1)
precipitation = np.ma.masked_invalid(precipitation)
precipitation2 = np.ma.mean(precipitation,axis = 1)
psl_arctic = np.ma.masked_invalid(psl_arctic)
psl_arctic2 = np.ma.mean(psl_arctic,axis = 1)
psl_spg = np.ma.masked_invalid(psl_spg)
psl_spg2 = np.ma.mean(psl_spg,axis = 1)
psl_diff = np.ma.masked_invalid(psl_diff)
psl_diff2 = np.ma.mean(psl_diff,axis = 1)
evap2 = np.ma.mean(evap,axis = 1)


#Arctic sea ice extent and more...

west = -24
east = -5
south = 65
north = 81

models = ['MRI-CGCM3', 'bcc-csm1-1', 'MPI-ESM-P', 'GISS-E2-R', 'CSIRO-Mk3L-1-2', 'HadCM3', 'MIROC-ESM', 'CCSM4']


directory = '/data/NAS-ph290/ph290/cmip5/last1000/'
data_and_forcings = {}

for model in models:
	data_and_forcings[model] = {}
		###########################
		#   Whole Arctic sea ice  #
		###########################
	cube1 = iris.load_cube(directory+model+'_sic_past1000_r1i1p1*.nc')
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
		cube1 = iris.load_cube(directory+model+'_sfcWind_past1000_r1i1p1*.nc')
		cube1 = extract_years(cube1)
	except:
		#2) does model have near surface u and v winds (uas, vas)?
		try:
			cube1 = extract_years(iris.load_cube(directory+model+'_uas_past1000_r1i1p1*.nc'))
			cube1_b = extract_years(iris.load_cube(directory+model+'_vas_past1000_r1i1p1*.nc'))
			cube1 = ((cube1*cube1) + (cube1_b*cube1_b))**(.5)
		except:
			#3) does model have 3D winds (ua and va)?
			cube1 = iris.load_cube(directory+model+'_ua_past1000_r1i1p1*.nc')
			cube1 = cube1.extract(iris.Constraint(str(cube1.coord(dimensions=1).standard_name)+' = '+str(np.max(cube1.coord(dimensions=1).points))))
			cube1 = extract_years(cube1)
			cube1_b = extract_years(iris.load_cube(directory+model+'_va_past1000_r1i1p1*.nc'))
			cube1_b = cube1_b.extract(iris.Constraint(str(cube1_b.coord(dimensions=1).standard_name)+' = '+str(np.max(cube1_b.coord(dimensions=1).points))))
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
		cube1 = iris.load_cube(directory+model+'_hurs_past1000_r1i1p1*.nc')
		cube1 = extract_years(cube1)
	except:
		#3) does model have 3D humidity (hur)?
		cube1 = iris.load_cube(directory+model+'_hur_past1000_r1i1p1*.nc')
		cube1 = cube1.extract(iris.Constraint(str(cube1.coord(dimensions=1).standard_name)+' = '+str(np.max(cube1.coord(dimensions=1).points))))
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
	cube1 = iris.load_cube(directory+model+'_rsds_past1000_r1i1p1*.nc')
	cube1 = extract_years(cube1)
	cube1b = iris.load_cube(directory+model+'_rsus_past1000_r1i1p1*.nc')
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
	cube1 = iris.load_cube(directory+model+'_rsdt_past1000_r1i1p1*.nc')
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
	cube1 = iris.load_cube(directory+model+'_tas_past1000_r1i1p1*.nc')
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
		cube1 = iris.load_cube(directory+model+'_sos_past1000_r1i1p1*.nc')
		cube1 = extract_years(cube1)
	except:
		#3) does model have 3D salinity (so)?
		cube1 = iris.load_cube(directory+model+'_so_past1000_r1i1p1*.nc')
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
	cube1 = iris.load_cube(directory+model+'_evspsbl_past1000_r1i1p1*.nc')
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
#     as described in Eq 32 of Valiantzas Journal of Hydrology (2006) 331            #
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

for model in models:
	print 'note: models may change volcanic forcing differently such that TOA radiation does not include volcanoc reductions - a further variable exists in CMIP5 if required'
	day_sec = 60.0 * 60.0 * 24.0
	T = data_and_forcings[model]['tas_n_iceland'] - 273.15
	Rs = data_and_forcings[model]['net_surface_downward_sw_n_iceland'] * day_sec * 1.9e-6
	Ra = data_and_forcings[model]['net_TOA_downward_sw_n_iceland'] * day_sec * 1.9e-6
	RH = data_and_forcings[model]['humidity_n_iceland']
	u = data_and_forcings[model]['windspeed_n_iceland']
	au = 0.0
	sea_ice_fraction = data_and_forcings[model]['ice_area_n_iceland']
	data_and_forcings[model]['reconstructed_evap'] = 0.051 * Rs * (T + 9.5)**(0.5) - 2.4 * (Rs/Ra) * (Rs/Ra) + 0.052 * (T + 20.0) * (1.0 - (RH/100.0)) * (au - 0.38 + 0.54 * u)
	data_and_forcings[model]['reconstructed_evap_multiplied_by_seaice'] = data_and_forcings[model]['reconstructed_evap'] * sea_ice_fraction
	T_temp = T.copy()
	T_temp = T_temp * 0.0 +np.mean(T)
	data_and_forcings[model]['reconstructed_evap_T_const'] = 0.051 * Rs * (T_temp + 9.5)**(0.5) - 2.4 * (Rs/Ra) * (Rs/Ra) + 0.052 * (T_temp + 20.0) * (1.0 - (RH/100.0)) * (au - 0.38 + 0.54 * u)
	u_temp = u.copy()
	u_temp = u_temp * 0.0 +np.mean(u)
	data_and_forcings[model]['reconstructed_evap_wind_const'] = 0.051 * Rs * (T + 9.5)**(0.5) - 2.4 * (Rs/Ra) * (Rs/Ra) + 0.052 * (T + 20.0) * (1.0 - (RH/100.0)) * (au - 0.38 + 0.54 * u_temp)
	RH_temp = RH.copy()
	RH_temp = RH_temp * 0.0 +np.mean(RH)
	data_and_forcings[model]['reconstructed_evap_humidity_const'] = 0.051 * Rs * (T + 9.5)**(0.5) - 2.4 * (Rs/Ra) * (Rs/Ra) + 0.052 * (T + 20.0) * (1.0 - (RH_temp/100.0)) * (au - 0.38 + 0.54 * u)
	Rs_temp = Rs.copy()
	Rs_temp = Rs_temp * 0.0 +np.mean(Rs)
	data_and_forcings[model]['reconstructed_evap_surface_radiation_const'] = 0.051 * Rs_temp * (T + 9.5)**(0.5) - 2.4 * (Rs_temp/Ra) * (Rs_temp/Ra) + 0.052 * (T + 20.0) * (1.0 - (RH/100.0)) * (au - 0.38 + 0.54 * u)
	Ra_temp = Ra.copy()
	Ra_temp = Ra_temp * 0.0 +np.mean(Ra)
	evap = 0.051 * Rs * (T + 9.5)**(0.5) - 2.4 * (Rs/Ra_temp) * (Rs/Ra_temp) + 0.052 * (T + 20.0) * (1.0 - (RH/100.0)) * (au - 0.38 + 0.54 * u)
	#convert mm/day = kg/m2/s
	evap = evap / day_sec
	data_and_forcings[model]['reconstructed_evap_TOA_radiation_const'] = evap



'''

#Add the model-specific volcanic solar and forcing data to the dictionary
for model in models:
	year = data_and_forcings[model]['year']
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


r_data_file = '/home/ph290/data0/reynolds/ultra_data.csv'
r_data = np.genfromtxt(r_data_file,skip_header = 1,delimiter = ',')
tmp = r_data[:,1]
tmp = scipy.signal.filtfilt(b, a, tmp)
tmp = rm.running_mean(tmp,smoothing_val)
loc = np.where((np.logical_not(np.isnan(tmp))) & (r_data[:,0] >= start_date) & (r_data[:,0] <= end_date))
tmp = tmp[loc]
tmp_yr = r_data[loc[0],0]



model = models[0]
ice_area_n_iceland_all = np.empty([np.size(data_and_forcings[model]['year']),np.size(models)])
forcing_all = np.empty([np.size(data_and_forcings[model]['year']),np.size(models)])

smoothing_val = 20

for i,model in enumerate(models):
    plt.close('all')
    fig, ax1 = plt.subplots()
    x = data_and_forcings[model]['year']
    y1 = data_and_forcings[model]['solar']
    y1 = scipy.signal.filtfilt(b, a, y1)
    y1 = rm.running_mean(y1,smoothing_val)
    y2 = data_and_forcings[model]['ice_area_n_iceland']
    y2 = scipy.signal.filtfilt(b, a, y2)
    y2 = rm.running_mean(y2,smoothing_val)
    y3 = data_and_forcings[model]['volc']
    y3=(y3-np.nanmin(y3))/(np.nanmax(y3)-np.nanmin(y3))
    #y3 = rmp.running_mean_post(y3,20)
    ################################
    y3 = rm.running_mean(y3,20)
    y3 = np.log(y3+1)
    y3=(y3-np.nanmin(y3))/(np.nanmax(y3)-np.nanmin(y3))
    #y3 = scipy.signal.filtfilt(b, a, y3)
    #y3 = np.log(y3+1)
    #y3=(y3-np.nanmin(y3))/(np.nanmax(y3)-np.nanmin(y3))
    plotting_y = (y1*-1.0)+y3*1.5
    ax1.plot(x,plotting_y,'g')
    #ax1.plot(x,(y1*1.0),'g.')
    #ax1.plot(x,y3*2.0,'g--')
    ax2 = ax1.twinx()
    ax2.plot(x,y2,'b')
    ################################
    #ax3 = ax2.twinx()
    #ax3.plot(x,y3,'r')
    # y3b = np.log(y3+1)
    # y3b[np.where(np.logical_not(np.isfinite(y3b)))] = 0
    # y1[np.where(np.logical_not(np.isfinite(y1)))] = 0
    # y = np.column_stack((y1,y3b))
    # y = sm.add_constant(y)
    # y2[np.where(np.logical_not(np.isfinite(y2)))] = 0
    # mlr_model = sm.OLS(y2,y)
    # results = mlr_model.fit()
    # ax1.plot(x,y2)
    # ax1.plot(x,results.params[2]*y3b+results.params[1]*y1+results.params[0])
    #ax1.scatter((y1*-1.0)+y3,y2)
    plt.show(block = False)
    ice_area_n_iceland_all[0:np.size(y2),i] = y2
    forcing_all[0:np.size(plotting_y),i] = plotting_y


ice_area_n_iceland_mean = scipy.stats.nanmean(ice_area_n_iceland_all,axis = 1)
forcing_mean = scipy.stats.nanmean(forcing_all,axis = 1)

plt.close('all')
fig, ax1 = plt.subplots()
ax1.plot(range(850,1851),forcing_mean,'w')
ax2 = ax1.twinx()
ax2.plot(range(850,1851),ice_area_n_iceland_mean,'b')
#ax3 = ax2.twinx()
#ax3.plot(tmp_yr,tmp,'g')
plt.show()

'''
This seems to be working.... Average across ensmble and hopefully we have the answer!

Maybe look at other significant forcings - labduse (more tricky...)?
'''

'''

arctic_sea_ice_area_all = {}

for model in models:
	print model
	cube1 = iris.load_cube(directory+model+'_sic_past1000_r1i1p1*.nc')
	cube1 = extract_years(cube1)
	arctic_sea_ice_area_all[model] = {}
	temporary_cube = cube1.intersection(longitude = (0, 360))
	arctic_sea_ice_fraction = temporary_cube.intersection(latitude = (0, 90))
	try:
		arctic_sea_ice_fraction.coord('latitude').guess_bounds()
		arctic_sea_ice_fraction.coord('longitude').guess_bounds()
	except:
		'has bounds'
	grid_areas = iris.analysis.cartography.area_weights(arctic_sea_ice_fraction)
	arctic_sea_ice_area = arctic_sea_ice_fraction.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	arctic_sea_ice_area_all[model]['ice_area'] = arctic_sea_ice_area.data
	coord = arctic_sea_ice_area.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	arctic_sea_ice_area_all[model]['year'] = year


plt.close('all')
for model in models:
	x = arctic_sea_ice_area_all[model]['year']
	y = arctic_sea_ice_area_all[model]['ice_area'].data
	#y = (y-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))
	y = signal.detrend(y)
	y = rm.running_mean(y,20)
	plt.plot(x,y)
	

plt.show(block = False)
'''
