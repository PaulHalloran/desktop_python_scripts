import numpy as np
import iris 
import matplotlib.pyplot as plt
import seawater
import glob
import os
import pickle
import numpy.ma as ma
import running_mean as rm
import running_mean_post as rmp
from scipy import signal
import scipy
import scipy.stats
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d
import cartopy.crs as ccrs
import iris.plot as iplt
import cartopy.feature as cfeature


'''

N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
#low_cutoff=150.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
#high_cutoff=150.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
#middle_cuttoff_low=1.0
################
middle_cuttoff_high=100.0
################

#Wn_low=timestep_between_values/low_cutoff
#Wn_high=timestep_between_values/high_cutoff
#Wn_mid_low=timestep_between_values/middle_cuttoff_low
Wn_mid_high=timestep_between_values/middle_cuttoff_high

#b, a = scipy.signal.butter(N, Wn_low, btype='low')
#b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')
#data = scipy.signal.filtfilt(b2, a2, tmp)



directory = '/media/usb_external1/cmip5/last1000/'

def model_names_tos(directory):
	files = glob.glob(directory+'/*tos*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models

def model_names_sos(directory):
        files = glob.glob(directory+'/*sos*.nc')
        models_tmp = []
        for file in files:
                statinfo = os.stat(file)
                if statinfo.st_size >= 1:
                        models_tmp.append(file.split('/')[-1].split('_')[0])
                        models = np.unique(models_tmp)
        return models

def model_names_pr(directory):
        files = glob.glob(directory+'/*pr*.nc')
        models_tmp = []
        for file in files:
                statinfo = os.stat(file)
                if statinfo.st_size >= 1:
                        models_tmp.append(file.split('/')[-1].split('_')[0])
                        models = np.unique(models_tmp)
        return models


models_I = model_names_tos(directory)
models_II = model_names_sos(directory)
models_III = model_names_pr(directory)

tmp = list(set(models_I).intersection(models_II))
models = np.array(list(set(tmp).intersection(models_III)))

west = -24
east = -13
south = 65
north = 67

density_data = {}

for model in models:
	print model
# 	try:
	#temperature
	t_cube = iris.load_cube(directory+model+'_tos_past1000_r*_regridded_not_vertically.nc')
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
	s_cube = iris.load_cube(directory+model+'_sos_past1000_r*_regridded_not_vertically.nc')
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
	pr_cube = iris.load_cube(directory+model+'_pr_past1000_r*_regridded.nc')
	try:
		pr_depths = pr_cube.coord('depth').points
		pr_cube = pr_cube.extract(iris.Constraint(depth = np.min(pr_depths)))
	except:
		print 'no precipitatoin depth coordinate'
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
	tmp_temp_mean_density = seawater.dens(s_cube_n_iceland_mean,t_cube_n_iceland_mean*0.0+np.mean(t_cube_n_iceland_mean)-273.15)
	tmp_sal_mean_density = seawater.dens(s_cube_n_iceland_mean*0.0+np.mean(s_cube_n_iceland_mean), t_cube_n_iceland_mean-273.15)
	#years
	coord = t_cube_n_iceland.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	#output
	density_data[model] = {}
	density_data[model]['temperature'] = t_cube_n_iceland_mean
	density_data[model]['salinity'] = s_cube_n_iceland_mean
	density_data[model]['density'] = tmp_density
	density_data[model]['temperature_meaned_density'] = tmp_temp_mean_density
	density_data[model]['salinity_meaned_density'] = tmp_sal_mean_density
	density_data[model]['precipitation'] = pr_cube_n_iceland_mean
	density_data[model]['years'] = year_tmp
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

years = range(min_yr,max_yr+1)
for i,model in enumerate(density_data.viewkeys()):
	tmp_yrs = density_data[model]['years']
	data1 = density_data[model]['density']
	data1 = scipy.signal.filtfilt(b2, a2, data1)
	data2 = density_data[model]['temperature']
	data2 = scipy.signal.filtfilt(b2, a2, data2)
	data3 = density_data[model]['salinity']
	data3 = scipy.signal.filtfilt(b2, a2, data3)
	data4 = density_data[model]['temperature_meaned_density']
	data4 = scipy.signal.filtfilt(b2, a2, data4)
	data5 = density_data[model]['salinity_meaned_density']
	data5 = scipy.signal.filtfilt(b2, a2, data5)
	data6 = density_data[model]['precipitation']
	data6 = scipy.signal.filtfilt(b2, a2, data6)
	for j,tmp_yr in enumerate(tmp_yrs):
		loc = np.where(tmp_yr == years)
		mean_density[loc,i] = data1[j]
		mean_temperature[loc,i] = data2[j]
		mean_salinity[loc,i] = data3[j]
		temperature_meaned_density[loc,i] = data4[j]
		salinity_meaned_density[loc,i] = data5[j]
		precipitation[loc,i] = data6[j]

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


###
#Now read in AMOC data
###


start_year = 850
end_year = 1850

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo.pickle', 'w') as f:
#    pickle.dump([all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,solar_I,solar_II#,solar_III], f)

with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo.pickle') as f:
    all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,solar_I,solar_II,solar_III = pickle.load(f)

all_models = ['bcc-csm1-1', 'MRI-CGCM3','CCSM4','GISS-E2-R', 'MIROC-ESM','MPI-ESM-P','HadCM3','CSIRO-Mk3L-1-2']

solar_I = ['CCSM4','FGOALS-gl', 'FGOALS-s2', 'MPI-ESM-P','bcc-csm1-1']
solar_II = ['GISS-E2-R','HadCM3','CSIRO-Mk3L-1-2']
solar_III = ['MRI-CGCM3','MIROC-ESM']
solar_not = ['CCSM4','MPI-ESM-P','bcc-csm1-1','MRI-CGCM3','MIROC-ESM']

volc_I = ['bcc-csm1-1', 'FGOALS-gl', 'MRI-CGCM3','CCSM4']
volc_II = ['GISS-E2-R', 'MIROC-ESM','MPI-ESM-P','HadCM3','CSIRO-Mk3L-1-2']


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


################
#calculate means.........
################

all_years = np.linspace(850,1850,(1851-850))
average_tas_solar_III = np.empty([np.size(all_years),np.size(solar_III)])
average_tas_solar_III[:] = np.NAN
average_tas_solar_II = np.empty([np.size(all_years),np.size(solar_II)])
average_tas_solar_II[:] = np.NAN
average_tas_solar_I = np.empty([np.size(all_years),np.size(solar_I)])
average_tas_solar_I[:] = np.NAN

average_str_solar_III = np.empty([np.size(all_years),np.size(solar_III)])
average_str_solar_III[:] = np.NAN
average_str_solar_II = np.empty([np.size(all_years),np.size(solar_II)])
average_str_solar_II[:] = np.NAN
average_str_solar_I = np.empty([np.size(all_years),np.size(solar_I)])
average_str_solar_I[:] = np.NAN

average_tas_volc_II = np.empty([np.size(all_years),np.size(volc_II)])
average_tas_volc_II[:] = np.NAN
average_tas_volc_I = np.empty([np.size(all_years),np.size(volc_I)])
average_tas_volc_I[:] = np.NAN

average_str_volc_II = np.empty([np.size(all_years),np.size(volc_II)])
average_str_volc_II[:] = np.NAN
average_str_volc_I = np.empty([np.size(all_years),np.size(volc_I)])
average_str_volc_I[:] = np.NAN

smoothing_val = 10

counter = -1
for i,model in enumerate(solar_III):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_solar_III[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_solar_III[j,counter] = data[loc2]
		except:
			None


counter = -1
for i,model in enumerate(solar_II):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_solar_II[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_solar_II[j,counter] = data[loc2]
		except:
			None


counter = -1
for i,model in enumerate(solar_I):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_solar_I[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_solar_I[j,counter] = data[loc2]
		except:
			None

counter = -1
for i,model in enumerate(volc_I):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_volc_I[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_volc_I[j,counter] = data[loc2]
		except:
			None

counter = -1
for i,model in enumerate(volc_II):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_volc_II[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_volc_II[j,counter] = data[loc2]
		except:
			None


average_tas_solar_IIIb = np.mean(average_tas_solar_III,axis = 1)
average_tas_solar_IIb = np.mean(average_tas_solar_II,axis = 1)
average_tas_solar_Ib = np.mean(average_tas_solar_I,axis = 1)

average_str_solar_IIIb = np.mean(average_str_solar_III,axis = 1)
average_str_solar_IIb = np.mean(average_str_solar_II,axis = 1)
average_str_solar_Ib = np.mean(average_str_solar_I,axis = 1)

average_tas_volc_IIb = np.mean(average_tas_volc_II,axis = 1)
average_tas_volc_Ib = np.mean(average_tas_volc_I,axis = 1)

average_str_volc_IIb = np.mean(average_str_volc_II,axis = 1)
average_str_volc_Ib = np.mean(average_str_volc_I,axis = 1)

#plotting means

'''

'''

mean_data_str = np.zeros([1+end_year-start_year,np.size(all_models)])
mean_data_str[:] = np.NAN
mean_data_str_subset = np.zeros([1+end_year-start_year,np.size(solar_II)])
mean_data_str_subset[:] = np.NAN

mean_data_str_subsetb = np.zeros([1+end_year-start_year,np.size(solar_I)])
mean_data_str_subsetb[:] = np.NAN

mean_data_str_subsetc = np.zeros([1+end_year-start_year,np.size(solar_III)])
mean_data_str_subsetc[:] = np.NAN

for i,model in enumerate(all_models):
	print model
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	for j,yr_tmp in enumerate(range(start_year,end_year)):
		loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
		if np.size(loc2) > 0:
			mean_data_str[j,i] = data3[loc2]
	
mean_data_str2 = np.mean(mean_data_str, axis = 1)

for i,model in enumerate(solar_II):
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs_subset = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	for j,yr_tmp in enumerate(range(start_year,end_year)):
		loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
		if np.size(loc2) > 0:
			mean_data_str_subset[j,i] = data3[loc2]

mean_data_str_subset2 = np.mean(mean_data_str_subset, axis = 1)

for i,model in enumerate(solar_I):
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs_subset = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	for j,yr_tmp in enumerate(range(start_year,end_year)):
		loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
		if np.size(loc2) > 0:
			mean_data_str_subsetb[j,i] = data3[loc2]

mean_data_str_subset2b = np.mean(mean_data_str_subsetb, axis = 1)


for i,model in enumerate(solar_III):
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs_subset = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	for j,yr_tmp in enumerate(range(start_year,end_year)):
		loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
		if np.size(loc2) > 0:
			mean_data_str_subsetc[j,i] = data3[loc2]

mean_data_str_subset2c = np.mean(mean_data_str_subsetc, axis = 1)



smoothing_val = 5
alph = 0.2
wdth = 2

#solar tas
plt.close('all')
fig = plt.figure(figsize=(10,10))

start_date = 850
end_date = 1850

ax31 = fig.add_subplot(411)

r_data_file = '/home/ph290/data0/reynolds/ultra_data.csv'
r_data = np.genfromtxt(r_data_file,skip_header = 1,delimiter = ',')

tmp = r_data[:,1]
tmp = scipy.signal.filtfilt(b2, a2, tmp)

smoothing_val = 5

tmp = rm.running_mean(tmp,smoothing_val)
loc = np.where((np.logical_not(np.isnan(tmp))) & (r_data[:,0] >= start_date) & (r_data[:,0] <= end_date))
tmp = tmp[loc]
tmp_yr = r_data[loc[0],0]
tmp = scipy.signal.filtfilt(b2, a2, tmp)

l4a = ax31.plot(tmp_yr,tmp,'r',linewidth = 2,alpha = 0.75,label = 'Reynolds d18O')

#ax41.plot(volc_yr_II,results.params[2]*x2+results.params[1]*x1+results.params[0],'r',linewidth = 3,alpha = 0.75)


ax32 = ax31.twinx()
tmp2 = rm.running_mean(mean_data_str2,smoothing_val)
#loc = np.where((np.logical_not(np.isnan(tmp2))) & (yrs >= start_date) & (yrs <= end_date))
#tmp2 = tmp2[loc]
tmp_yr = yrs
#ax42.plot(tmp_yr,tmp2,'g',linewidth = 2,alpha = 0.75)

l4b = ax32.plot(yrs,rm.running_mean(mean_data_str2,smoothing_val),'g',linewidth = 2,alpha = 0.75,label = 'CMIP5/PMIP3 ensemble mean AMOC')
ax32.set_ylim([0.3,0.5])
ax31.set_xlim([850,1850])

ax31.set_ylabel('Normalised\nAMOC strength')
ax32.set_ylabel('d$^{18}$O')

# for i,model in enumerate(density_data.viewkeys()):
# 	plt.close('all')
# 	fig = plt.figure(figsize=(10,5))
# 	ax11 = fig.add_subplot(111)
# 	ax11.plot(density_data[model]['years'],density_data[model]['temperature'],'r')
# 	ax12 = ax11.twinx()
# 	ax12.plot(density_data[model]['years'],density_data[model]['salinity'],'b')
# 	plt.show()
	
#temperature and salinity seem to really closely co-vary here..


lns = l4a+l4b
 
#fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})

###
#amoc and density
###

ax11 = fig.add_subplot(412)


for i,dummy in enumerate(all_models):
	l1 = ax11.plot(range(start_year,end_year+1),rm.running_mean(mean_data_str[:,i],smoothing_val),'g',alpha = 0.1,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble member AMOC')

l2 = ax11.plot(range(start_year,end_year+1),rm.running_mean(mean_data_str2,smoothing_val),'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean AMOC')

ax11.set_ylim([0.3,0.5])
ax11.set_ylabel('Normalised\nAMOC strength')


# amo_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
# amo = np.genfromtxt(amo_file, skip_header = 4)
# amo_yr = amo[:,0]
# amo_data = amo[:,1]
# loc = np.where((amo_yr <= 1850) & (amo_yr >= 850))
# amo_yr = amo_yr[loc]
# amo_data = amo_data[loc]
# amo_data = scipy.signal.filtfilt(b2, a2, amo_data)
# 
ax12 = ax11.twinx()
l3 = ax12.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')
# ax12.set_ylim([-0.4,0.3])
ax12.set_ylabel('Density anomaly\n(kg/m$^3$)')

lns = l1+l2+l3
 
#fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})


ax21 = fig.add_subplot(413)
l3 = ax21.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland density - note filtering poss. messing up ends')
l3b = ax21.plot(years,rm.running_mean(temperature_meaned_density2,smoothing_val),'r',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to salinity)')
l3c = ax21.plot(years,rm.running_mean(salinity_meaned_density2,smoothing_val),'b',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to temperature)')
# ax21.set_ylim([-400,300])

ax21.set_ylabel('Density anomaly\n(kg/m$^3$)')

lns = l3+l3b+l3c
 
#fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})



###
#Precipitatoin driving salinity?
###


#wdth = 2

ax41 = fig.add_subplot(414)
l5a = ax41.plot(years,rm.running_mean(mean_salinity2,smoothing_val),'b',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland salinity')
ax42=ax41.twinx()
l5b = ax42.plot(years,rm.running_mean(precipitation2,smoothing_val)/1.0e6,'r',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland precipitation - note signal processing seems to have flipped sign of salinity')


ax43=ax41.twinx()
l5c = ax43.plot(voln_n[:,0],voln_n[:,1],'k',linewidth = wdth,alpha = 0.2,label = 'volcanic index')
#ax43.plot(voln_n[:,0],rmp.running_mean_post(voln_n[:,1],36*7)*20,'k',alpha = 0.5,label = 'volcanic index')

ax41.set_ylabel('salinity anomaly')
ax42.set_ylabel('precipitation anomaly x10$^{-6}$')
ax41.set_xlabel('Calendar Year')

ax41.set_xlim([850,1850])
ax42.set_xlim([850,1850])
ax43.set_xlim([850,1850])

#ax42.axis('off')
ax43.axis('off')

lns = l5a+l5b+l5c
 
#fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})



'''

# lns = l1+l2+l3+l3b+l3c+l4a+l4b+l5a+l5b+l5c
 
# #fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})

'''

ax11.set_xlim([850,1850])
ax21.set_xlim([850,1850])
plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/palaeoamo/AMOC_and_n_aceland_density.png')

'''
'''

#calculate density with t and s held fixed...


###
#plotting temperature, salinity, density relationships
###

wdth = 2

plt.close('all')
fig = plt.figure(figsize=(10,5))
ax11 = fig.add_subplot(211)

ax11.plot(years,rm.running_mean(mean_temperature2,smoothing_val),'r',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland temperature')
ax12 = ax11.twinx()
ax12.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density')

ax21 = fig.add_subplot(212)
ax21.plot(years,rm.running_mean(mean_salinity2,smoothing_val),'b',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland salinity')
ax22 = ax11.twinx()
ax22.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density')

plt.show(block = True)


###############


for i,model in enumerate(density_data.viewkeys()):
	plt.close('all')
	fig = plt.figure(figsize=(10,5))
	ax11 = fig.add_subplot(111)
	ax11.plot(density_data[model]['years'],density_data[model]['temperature'],'r')
	ax12 = ax11.twinx()
	ax12.plot(density_data[model]['years'],density_data[model]['salinity'],'b')
	plt.show()
	
#temperature and salinity seem to really closely co-vary here..


###
#Precipitatoin driving salinity?
###


wdth = 2

plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(111)
l1 = ax1.plot(years,rm.running_mean(mean_salinity2,smoothing_val),'b',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland salinity')
ax2=ax1.twinx()
l2 = ax2.plot(years,rm.running_mean(precipitation2,smoothing_val),'r',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland precipitation - note signal processing seems to have flipped sign of saliiity')


ax3=ax2.twinx()
l3 = ax3.plot(voln_n[:,0],voln_n[:,1],'k',alpha = 0.5,label = 'volcanic index')
ax3.plot(voln_n[:,0],rmp.running_mean_post(voln_n[:,1],36*7)*20,'k',alpha = 0.5,label = 'volcanic index')

lns = l1+l2+l3
 
#fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'upper left', fancybox=True, framealpha=0.2,prop={'size':8})


ax1.set_xlim([850,1850])
ax2.set_xlim([850,1850])
ax3.set_xlim([850,1850])

plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/palaeoamo/AMOC_and_n_iceland_sal_precip.png')



'''

###
#era interim nao precipitatoin
###

file = '/data/temp/ph290/era_interim/era_moisture_flux_ann_mean.nc'
era_cube = iris.load_cube(file)
era_cube = era_cube[:-1]
coord = era_cube.coord('time')
dt = coord.units.num2date(coord.points)
era_year = np.array([coord.units.num2date(value).year for value in coord.points])

nao = np.genfromtxt('/home/ph290/data0/misc_data/norm.nao.monthly.b5001.current.ascii.table',skip_footer = 1)
#http://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/JFM_season_nao_index.shtml
nao_year = nao[:,0]
jfm_nao_data = np.mean(nao[:,1:4],axis = 1)
loc = np.where(nao_year >= np.min(era_year))
nao_year = nao_year[loc]
jfm_nao_data = jfm_nao_data[loc]
jfm_nao_data = jfm_nao_data-np.mean(jfm_nao_data)

loc2_low = np.where(jfm_nao_data < 0)
loc2_high = np.where(jfm_nao_data > 0)

era_cube_low = era_cube[loc2_low].collapsed('time',iris.analysis.MEAN)
era_cube_high = era_cube[loc2_high].collapsed('time',iris.analysis.MEAN)

west = -180
east = 180
south = 20
north = 90

temporary_cube = era_cube_low.intersection(longitude = (west, east))
era_cube_low = temporary_cube.intersection(latitude = (south, north))
temporary_cube = era_cube_high.intersection(longitude = (west, east))
era_cube_high = temporary_cube.intersection(latitude = (south, north))


###
#Looking at pattern of precipitation change
###

years = np.array(years)
loc = np.where(mean_salinity2[100:-100] > 0.0)
tmp_years = years[100:-100]
high_years = tmp_years[loc[0]]

loc = np.where(precipitation2[100:-100] < 0.0)
tmp_years = years[100:-100]
low_years = tmp_years[loc[0]]

#read in precip from each of the models
#low-pass filter the data
#produce high/loc composites...


#

#density_data = {}

pr_high = np.zeros([models.size,180,360])
pr_high[:] = np.NAN
pr_low = pr_high.copy()

for i,model in enumerate(models):
	print model
	pr_cube = iris.load_cube(directory+model+'_pr_past1000_r*_regridded.nc')
	try:
		pr_depths = pr_cube.coord('depth').points
		pr_cube = pr_cube.extract(iris.Constraint(depth = np.min(pr_depths)))
	except:
		print 'no precipitatoin depth coordinate'
	pr_cube.data = scipy.signal.filtfilt(b2, a2, pr_cube.data,axis = 0)
	# years
	coord = t_cube_n_iceland.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	common_high_years = np.array(list(set(year_tmp).intersection(high_years)))
	high_yr_index = np.nonzero(np.in1d(year_tmp,common_high_years))[0]
	common_low_years = np.array(list(set(year_tmp).intersection(low_years)))
	low_yr_index = np.nonzero(np.in1d(year_tmp,common_low_years))[0]
	pr_cube_high = pr_cube[high_yr_index].collapsed('time',iris.analysis.MEAN)
	pr_cube_low = pr_cube[low_yr_index].collapsed(['time'],iris.analysis.MEAN)
	pr_high[i,:,:] = pr_cube_high.data
	pr_low[i,:,:] = pr_cube_low.data




pr_cube_high_mean = pr_cube_high.copy()
pr_cube_low_mean = pr_cube_low.copy()

pr_cube_high_mean.data = np.mean(pr_high,axis = 0)
pr_cube_low_mean.data = np.mean(pr_low,axis = 0)


west = -180
east = 180
south = 20
north = 90

temporary_cube = pr_cube_high_mean.intersection(longitude = (west, east))
pr_cube_high_mean = temporary_cube.intersection(latitude = (south, north))
temporary_cube = pr_cube_low_mean.intersection(longitude = (west, east))
pr_cube_low_mean = temporary_cube.intersection(latitude = (south, north))

###
#volcanic precipitation response
###


volc_yrs = voln_n[:,0]
voln_data = voln_n[:,1]

loc = np.where((volc_yrs >= 950) & (volc_yrs < 1750))
volc_yrs = volc_yrs[loc]
voln_data = voln_data[loc]
volc_yrs = np.floor(volc_yrs)
volc_yrs2 = np.unique(volc_yrs)

voln_data2 = volc_yrs2.copy()

for i,temp_yrs in enumerate(volc_yrs2):
	loc = np.where(volc_yrs == temp_yrs)
	voln_data2[i] = np.mean(voln_data[loc])

voln_data2 = rmp.running_mean_post(voln_data2,1)

loc = np.where(voln_data2 > np.median(voln_data2))
tmp_years = years[100:-100]
high_years = tmp_years[loc[0]]

loc = np.where(voln_data2 < np.median(voln_data2))
tmp_years = years[100:-100]
low_years = tmp_years[loc[0]]

pr_high_volc = np.zeros([models.size,180,360])
pr_high_volc[:] = np.NAN
pr_low_volc = pr_high_volc.copy()

for i,model in enumerate(models):
	print model
	pr_cube = iris.load_cube(directory+model+'_pr_past1000_r*_regridded.nc')
	try:
		pr_depths = pr_cube.coord('depth').points
		pr_cube = pr_cube.extract(iris.Constraint(depth = np.min(pr_depths)))
	except:
		print 'no precipitatoin depth coordinate'
	pr_cube.data = scipy.signal.filtfilt(b2, a2, pr_cube.data,axis = 0)
	# years
	coord = t_cube_n_iceland.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	common_high_years = np.array(list(set(year_tmp).intersection(high_years)))
	high_yr_index = np.nonzero(np.in1d(year_tmp,common_high_years))[0]
	common_low_years = np.array(list(set(year_tmp).intersection(low_years)))
	low_yr_index = np.nonzero(np.in1d(year_tmp,common_low_years))[0]
	pr_cube_high_volc = pr_cube[high_yr_index].collapsed('time',iris.analysis.MEAN)
	pr_cube_low_volc = pr_cube[low_yr_index].collapsed(['time'],iris.analysis.MEAN)
	pr_high_volc[i,:,:] = pr_cube_high_volc.data
	pr_low_volc[i,:,:] = pr_cube_low_volc.data


pr_cube_high_volc_mean = pr_cube_high_volc.copy()
pr_cube_low_volc_mean = pr_cube_low_volc.copy()

pr_cube_high_volc_mean.data = np.mean(pr_high_volc,axis = 0)
pr_cube_low_volc_mean.data = np.mean(pr_low_volc,axis = 0)


###
#plotting
###


plt.close('all')
fig = plt.figure(figsize = (20,10))
ax = plt.subplot(131,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
change_precip = pr_cube_low_mean-pr_cube_high_mean
my_plot = iplt.contourf(change_precip,np.linspace(-15.0e-7,15.0e-7,31),cmap='bwr')
ax.add_feature(cfeature.LAND,facecolor='#f6f6f6')
plt.gca().coastlines()
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(pr_cube_high_mean.long_name+' ('+format(pr_cube_high_mean.units)+')')
plt.title('PMIP3 high/low precip. composites\nrelating to high/low salinity years')

ax2 = plt.subplot(132,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax2.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
change_precip = pr_cube_low_volc_mean-pr_cube_high_volc_mean
my_plot = iplt.contourf(change_precip,np.linspace(-15.0e-7,15.0e-7,31),cmap='bwr')
ax2.add_feature(cfeature.LAND,facecolor='#f6f6f6')
plt.gca().coastlines()
#ax.add_feature(cfeature.RIVERS)
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(pr_cube_low_volc_mean.long_name+' ('+format(pr_cube_low_volc_mean.units)+')')
plt.title('PMIP3 precip:\nhigh/low volcanic year composites')

ax3 = plt.subplot(133,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax3.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
change_precip = era_cube_high-era_cube_low
my_plot = iplt.contourf(change_precip,np.linspace(-8.0e-6,8e-6,31),cmap='bwr')
ax3.add_feature(cfeature.LAND,facecolor='#f6f6f6')
plt.gca().coastlines()
#ax.add_feature(cfeature.RIVERS)
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(era_cube_high.long_name+' ('+format(era_cube_high.units)+')')
plt.title('ERA interim precip:\nhigh/low NAO composites')

plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/palaeoamo/precip_composites.png')



# years
# coord = t_cube_n_iceland.coord('time')
# dt = coord.units.num2date(coord.points)
# year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
# #output
# density_data[model] = {}
# density_data[model]['temperature'] = t_cube_n_iceland_mean
# density_data[model]['salinity'] = s_cube_n_iceland_mean
# density_data[model]['density'] = tmp_density
# density_data[model]['temperature_meaned_density'] = tmp_temp_mean_density
# density_data[model]['salinity_meaned_density'] = tmp_sal_mean_density
# density_data[model]['precipitation'] = pr_cube_n_iceland_mean
# density_data[model]['years'] = year_tmp	

