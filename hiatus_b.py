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
import matplotlib.cm as mpl_cm


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
directory2 = '/media/usb_external1/cmip5/pr_regridded/'
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
        file1 = glob.glob(directory+model+'_*'+'tas'+'*r1i1p1_*.nc')
        file2 = glob.glob(directory+model+'_*'+'tos'+'*r1i1p1_*.nc')
        file3 = glob.glob(directory2+model+'_*'+'pr_piControl*.nc')
        if ((np.size(file1) > 0) & (np.size(file2) > 0) & (np.size(file3) > 0)):
			print 'processing: '+model
			#tas
			file = glob.glob(directory+model+'_*'+'tas'+'*r1i1p1_*.nc')
			cube = iris.load_cube(file)
			file = glob.glob(directory+model+'_*'+'tos'+'*r1i1p1_*.nc')
			cube1 = iris.load_cube(file)
			file = glob.glob(directory2+model+'_*'+'pr_piControl_*.nc')
			cube2 = iris.load_cube(file)
			coord = cube1.coord('time')
			coord = cube.coord('time')
			dt = coord.units.num2date(coord.points)
			years = np.array([coord.units.num2date(value).year for value in coord.points])
			dt = coord.units.num2date(coord.points)
			years1 = np.array([coord.units.num2date(value).year for value in coord.points])
			coord = cube2.coord('time')
			dt = coord.units.num2date(coord.points)
			years2 = np.array([coord.units.num2date(value).year for value in coord.points])
			locs = np.in1d(years, years1)
			locs = np.in1d(years2,years[locs])
			cube = cube[locs]
			cube1 = cube1[locs]
			cube2 = cube2[locs]
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

			ts_filtered = ts_filtered[locs]
			ts_filtered_2D = cube.copy()
			ts_filtered_2D.data = np.swapaxes(np.swapaxes(np.tile(ts_filtered.data,[180,360,1]),1,2),0,1)
			cube.data = high_pass_filter(cube.data,upper_limit_years)
			cube.data = low_pass_filter(cube.data,lower_limit_years)
			data[model] = {}
			data[model]['tas'] = {}
			data[model]['tas'] = istats.pearsonr(ts_filtered_2D,cube,corr_coords=['time'])
			cube1.data = high_pass_filter(cube1.data,upper_limit_years)
			cube1.data = low_pass_filter(cube1.data,lower_limit_years)
			data[model]['tos'] = {}
			data[model]['tos'] = istats.pearsonr(ts_filtered_2D,cube1,corr_coords=['time'])
			cube2.data = high_pass_filter(cube2.data,upper_limit_years)
			cube2.data = low_pass_filter(cube2.data,lower_limit_years)
			data[model]['pr'] = {}
			data[model]['pr'] = istats.pearsonr(ts_filtered_2D,cube2,corr_coords=['time'])








models2 = list(data.viewkeys())
models2.remove('HadGEM2-ES')
no_models = np.size(models2)

tas_mean = data[models2[0]]['tas'].copy()
tas_stdev = tas_mean.copy()
tos_mean = data[models2[0]]['tos'].copy()
tos_stdev = tos_mean.copy()
pr_mean = data[models2[0]]['pr'].copy()
pr_stdev = pr_mean.copy()

data_tmp1 = np.empty([no_models,np.shape(data[models2[0]]['tos'].data)[0],np.shape(data[models2[0]]['tos'].data)[1]])*0.0+np.NAN
data_tmp2 = data_tmp1.copy()
data_tmp3 = data_tmp1.copy()


for i,model in enumerate(models2):
    data_tmp1[i,:,:] = data[model]['tas'].data
    data_tmp2[i,:,:] = data[model]['tos'].data
    data_tmp3[i,:,:] = data[model]['pr'].data


tas_mean.data = np.mean(data_tmp1,axis = 0)
tas_stdev.data = np.std(data_tmp1,axis = 0)
tos_mean.data = np.mean(data_tmp2,axis = 0)
tos_stdev.data = np.std(data_tmp2,axis = 0)
pr_mean.data = np.mean(data_tmp3,axis = 0)
pr_stdev.data = np.std(data_tmp3,axis = 0)

brewer_cmap = mpl_cm.get_cmap('brewer_RdBu_11')
plt.close('all')

figure(figsize=(12, 12)
fig, axes  = plt.subplots(nrows=2, ncols=3)
plt.subplot(3, 2, 1)
cs1 = iplt.contourf(tos_mean,np.linspace(-1,1,21), cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('SST (ens. mean)')

plt.subplot(3, 2, 2)
cs2 = iplt.contourf(tos_stdev,np.linspace(-1,1,21), cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('SST (ens. stdev)')

plt.subplot(3, 2, 3)
cs3 = iplt.contourf(tas_mean,np.linspace(-1,1,21), cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('surface T (ens. mean)')

plt.subplot(3, 2, 4)
cs4 = iplt.contourf(tas_stdev,np.linspace(-1,1,21), cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('surface T (ens. stdev)')

fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.85, 0.35, 0.02, 0.6])
cbar1 = plt.colorbar(cs4,orientation='vertical', cax=cbar_ax1)

plt.subplot(3, 2, 5)
cs5 = iplt.contourf(pr_mean,np.linspace(-0.5,0.5,21), cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('precipitation (ens. mean)')

plt.subplot(3, 2, 6)
cs6 = iplt.contourf(pr_stdev,np.linspace(-0.5,0.5,21), cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('precipitation (ens. stdev)')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.11, 0.02, 0.2])
cbar = plt.colorbar(cs6,orientation='vertical', cax=cbar_ax)

plt.savefig('/home/ph290/Documents/figures/hiatus_correlations.pdf')
#plt.show(block = False)




plt.close('all')
plt.subplot(5, 6, 1)

for i,model in enumerate(data.viewkeys()):
	plt.subplot(5, 6, i)
	iplt.contourf(data[model]['tos'],np.linspace(-1,1,21))
	plt.gca().coastlines()
	plt.title(model)

plt.savefig('/home/ph290/Documents/figures/tos_correlations_b.png')
#plt.show(block = False)

plt.close('all')
plt.subplot(5,6, 1)

for i,model in enumerate(data.viewkeys()):
	plt.subplot(5, 6, i)
	iplt.contourf(data[model]['tas'],np.linspace(-1,1,21))
	plt.gca().coastlines()
	plt.title(model)

plt.savefig('/home/ph290/Documents/figures/tas_correlations_b.png')
#plt.show(block = False)

plt.close('all')
plt.subplot(5, 6, 1)

for i,model in enumerate(data.viewkeys()):
	plt.subplot(5, 6, i)
	iplt.contourf(data[model]['pr'],np.linspace(-1,1,21))
	plt.gca().coastlines()
	plt.title(model)

plt.savefig('/home/ph290/Documents/figures/pr_correlations_b.png')
#plt.show(block = False)


