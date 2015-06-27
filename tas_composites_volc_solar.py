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
from eofs.iris import Eof

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

offset = 0
volc_threshold = 0.02
solar_threshold = 0 #because after filtering it varies around 0

data_and_forcings = {}

for model in models:
	print model
	data_and_forcings[model] = {}
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
	################################
	#   air temperature  #
	################################
	if str(cube1.coord(dimensions=1).standard_name) == 'depth':
		cube1 = cube1.extract(iris.Constraint(depth = 0))
	cube1b = cube1.copy()
	cube1b.data = scipy.signal.filtfilt(b, a, cube1.data,axis = 0)
	cube1b.data = scipy.signal.filtfilt(b2, a2,cube1b.data ,axis = 0)
	cube1b = extract_years(cube1b)
	################################
	#   airtas eofs                #
	################################
        #data_and_forcings[model]['tas_eof'] = Eof(cube1b)
	################################
	#   evaporation                #
	################################
	cube2 = iris.load_cube(directory+model+'_evspsbl_past1000_r1i1p1_*.nc')
	cube2 = extract_years(cube2)
	cube2b = cube2.copy()
	cube2b.data = scipy.signal.filtfilt(b, a, cube2.data,axis = 0)
	cube2b.data = scipy.signal.filtfilt(b2, a2,cube2b.data ,axis = 0)
	cube2b = extract_years(cube2b)
        #coord = cube2.coord('time')
	#dt = coord.units.num2date(coord.points)
	#year2 = np.array([coord.units.num2date(value).year for value in coord.points])
        #common_year = np.in1d(year,year2)
	################################
	#       volc composites        #
	################################
	ts_variable = data_and_forcings[model]['volc']
	#ts_variable = rm.running_mean(ts_variable,smoothing_val)
	min_loc = np.where(ts_variable <= volc_threshold)
	max_loc = np.where(ts_variable >= volc_threshold)    
	loc_min2 = np.in1d(year,year[min_loc]+offset)
	loc_max2 = np.in1d(year,year[max_loc]+offset)
	data_and_forcings[model]['tas_volc_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['tas_volc_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['evap_volc_composite_high'] = cube2b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['evap_volc_composite_low'] = cube2b[loc_min2].collapsed('time',iris.analysis.MEAN)
	################################
	#       solar composites       #
	################################
	ts_variable = data_and_forcings[model]['solar']
	ts_variable = scipy.signal.filtfilt(b, a, ts_variable,axis = 0)
	ts_variable = scipy.signal.filtfilt(b2, a2,ts_variable ,axis = 0)
	#ts_variable = rm.running_mean(ts_variable,smoothing_val)
	min_loc = np.where(ts_variable <= solar_threshold)
	max_loc = np.where(ts_variable >= solar_threshold)    
	loc_min2 = np.in1d(year,year[min_loc]+offset)
	loc_max2 = np.in1d(year,year[max_loc]+offset)
	data_and_forcings[model]['tas_solar_composite_high'] = cube1b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['tas_solar_composite_low'] = cube1b[loc_min2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['evap_solar_composite_high'] = cube2b[loc_max2].collapsed('time',iris.analysis.MEAN)
	data_and_forcings[model]['evap_solar_composite_low'] = cube2b[loc_min2].collapsed('time',iris.analysis.MEAN)



'''


for i,model in enumerate(models):
    plt.close('all')
    plt.figure(figsize = (20,6))
    solver = data_and_forcings[model]['tas_eof']
    pcs = solver.pcs(npcs=4, pcscaling=1)
    eofs = solver.eofs(neofs=4, eofscaling=1)
    no = 1
    y = pcs[:,no].data
    plt.plot(data_and_forcings[model]['year'],y/np.max(y),'b',linewidth = 3)
    y = model_forcing[model]['volc'][:,1]
    plt.plot(model_forcing[model]['volc'][:,0],(y/np.max(y))*(-4.0),'r')
    plt.plot(model_forcing[model]['volc'][:,0],(y/np.max(y))*(4.0),'r')
    y = model_forcing[model]['solar'][:,1]
    y = scipy.signal.filtfilt(b, a, y,axis = 0)
    y = scipy.signal.filtfilt(b2, a2, y ,axis = 0)
    plt.plot(model_forcing[model]['solar'][:,0],y/np.max(y)*(-1.0),'g')
    plt.plot(model_forcing[model]['solar'][:,0],y/np.max(y),'g--')
    plt.xlim(850,1850)
    plt.ylim(-3,3)
    print i
    plt.show()

model = models[3]
solver = data_and_forcings[model]['tas_eof']
eofs = solver.eofs(neofs=4, eofscaling=1)
qplt.contourf(eofs[1],31)
plt.gca().coastlines()
plt.show()

pc0
model0 = n/a
model1 = n/a
model2 = volc
model3 = n/a
model4 = solar????????
model5 = solar?
model6 = volc?
model7 = volc

pc1
model3 = volc and solar
model4 =  poss +ve volc

		
'''



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



volc_composite_mean_high = composite_mean(models,data_and_forcings,'tas_volc_composite_high')
volc_composite_mean_low = composite_mean(models,data_and_forcings,'tas_volc_composite_low')
solar_composite_mean_high = composite_mean(models,data_and_forcings,'tas_solar_composite_high')
solar_composite_mean_low = composite_mean(models,data_and_forcings,'tas_solar_composite_low')

volc_composite_mean_high_evap = composite_mean(models,data_and_forcings,'evap_volc_composite_high')
volc_composite_mean_low_evap = composite_mean(models,data_and_forcings,'evap_volc_composite_low')
solar_composite_mean_high_evap = composite_mean(models,data_and_forcings,'evap_solar_composite_high')
solar_composite_mean_low_evap = composite_mean(models,data_and_forcings,'evap_solar_composite_low')


plt.close('all')
qplt.contourf(volc_composite_mean_high,np.linspace(-0.3,0.3,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/volc_tas_composites_high.png')
# plt.show()

plt.close('all')
qplt.contourf(volc_composite_mean_low,np.linspace(-0.3,0.3,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/volc_tas_composites_low.png')
# plt.show()

plt.close('all')
qplt.contourf(solar_composite_mean_high,np.linspace(-0.2,0.2,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/solar_tas_composites_high.png')
# plt.show()

plt.close('all')
qplt.contourf(solar_composite_mean_low,np.linspace(-0.2,0.2,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/solar_tas_composites_low.png')
# plt.show()

###
# evap
###

plt.close('all')
qplt.contourf(volc_composite_mean_high_evap,np.linspace(-0.3e-6,0.3e-6,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/volc_evap_composites_high.png')
# plt.show()

plt.close('all')
qplt.contourf(volc_composite_mean_low_evap,np.linspace(-0.3e-6,0.3e-6,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/volc_evap_composites_low.png')
# plt.show()

plt.close('all')
qplt.contourf(solar_composite_mean_high_evap,np.linspace(-0.3e-6,0.3e-6,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/solar_evap_composites_high.png')
# plt.show()

plt.close('all')
qplt.contourf(solar_composite_mean_low_evap,np.linspace(-0.3e-6,0.3e-6,31))
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/solar_evap_composites_low.png')
# plt.show()

print 'NOW NEED TO PLAY AROUND WITH LAGS'
