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

'''
#producing an Atlantic mask (mask masked and Atlantic has value of 1, elsewhere zero)
'''

files = glob.glob('/media/usb_external1/cmip5/tas_regridded_summer/*_tas_past1000_*.nc')

models = []
ensembles = []
amo_box_tas = []
model_years_tas = []

for file in files:
	ensemble = file.split('/')[5].split('_')[3]
	ensembles.append(ensemble)
	if not 'r1i1p12' in ensemble:
		model = file.split('/')[5].split('_')[0]
		models.append(model)
		print 'processing: '+model
		#file = glob.glob('/media/usb_external1/cmip5/tas_regridded_winter/'+model+'_tas_past1000_'+ensemble+'_regridded.nc')
		#if not np.size(file) == 0:
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
loc = np.where(amo_yr <= 1850)
amo_yr = amo_yr[loc]
amo_data = amo_data[loc]
amo_data = signal.detrend(amo_data)

'''
#read in volc data
'''

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



###
#plotting
###


#AVERAGE TOGETHER!

smoothing_val = 5

all_years = np.linspace(850,1840,(1841-850))
average_tas = np.empty([np.size(all_years),np.size(models)-5])
average_tas[:] = np.NAN

counter = -1
for i,model in enumerate(models):
    if model not in ['xxxxxx']:
        counter += 1
        tmp = amo_box_tas[i].data
        tmp = rm.running_mean(tmp,smoothing_val)
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years_tas[i] <= 1840) & (model_years_tas[i] >= 850))
        tmp = tmp[loc]
        yrs = model_years_tas[i][loc]
        data = signal.detrend(tmp)
        for j,yr in enumerate(all_years):
                try:
                        loc2 = np.where(yrs == yr)
                        average_tas[j,counter] = data[loc2]
                except:
                        None
        

	
average_tas2 = np.mean(average_tas,axis = 1)

#wan_data = np.genfromtxt('/home/ph290/data0/misc_data/wanamaker_data.csv',skip_header = 1,delimiter = ',')


plt.close('all')
fig = plt.figure(figsize = (12,5))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(all_years,average_tas2,'r',linewidth=3,alpha = 0.5)
ax1.plot(amo_yr,signal.detrend(amo_data),'k',linewidth=3,alpha = 0.5)
ax3 = ax1.twinx()
ax3.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=3,alpha = 0.2)
ax3.plot(voln_s[:,0],voln_s[:,1],'b',linewidth=2,alpha = 0.2)
ax3.set_ylim([0,0.8])
#ax4 = ax3.twinx()
#ax4.scatter(wan_data[:,0],wan_data[:,1]*-1.0,color = 'red')
ax1.set_xlim([850,1850])
ax1.set_ylim(-0.5,0.5)

plt.tight_layout()
plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/amo_fig3summer.png')

