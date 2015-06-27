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
import pickle
'''

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
files = glob.glob(directory+'*tas*r1i1p1*.nc')

models = unique_models(files)
variables = ['tas']

###
#read in data
###


models = list(models)
# models.remove('CCSM4')
# models.remove('CMCC-CESM')
# models.remove('CMCC-CM')
# models.remove('CMCC-CMS')
# models.remove('CESM1-FASTCHEM')
# models.remove('HadGEM2-CC')
# #models.remove('MIROC-ESM')

models = np.array(models)

upper_limit_years = 30.0
lower_limit_years = 5.0

data = {}

for i,model in enumerate(models):
	print 'testing: '+model
        file1 = glob.glob(directory+model+'_*'+'tas'+'*r1i1p1_*.nc')
        file2 = glob.glob(directory+model+'_*'+'thetao'+'*r1i1p1_*.nc')
        if ((np.size(file1) > 0) & (np.size(file2) > 0)):
			print 'processing: '+model
			#tas
			file = glob.glob(directory+model+'_*'+'tas'+'*r1i1p1_*.nc')
			cube = iris.load_cube(file)
			file = glob.glob(directory+model+'_*'+'thetao'+'*r1i1p1_*.nc')
			cube1 = iris.load_cube(file)
			coord = cube1.coord('time')
			coord = cube.coord('time')
			dt = coord.units.num2date(coord.points)
			years = np.array([coord.units.num2date(value).year for value in coord.points])
			dt = coord.units.num2date(coord.points)
			years1 = np.array([coord.units.num2date(value).year for value in coord.points])
			locs = np.in1d(years, years1)
			if np.size(locs) > 500:
				locs = locs[0:500]
			cube = cube[locs]
			cube1 = cube1[locs]
			try:
					cube.coord('latitude').guess_bounds()
					cube.coord('longitude').guess_bounds()
			except:
					print 'already have bounds'		
			grid_areas = iris.analysis.cartography.area_weights(cube)
			ts_filtered = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
			ts_filtered.data = high_pass_filter(ts_filtered.data,upper_limit_years)
			ts_filtered.data = low_pass_filter(ts_filtered.data,lower_limit_years)	

			ts_filtered = ts_filtered[locs]
			ts_filtered_2D = cube1.copy()
			ts_filtered_2D.data = np.swapaxes(np.swapaxes(np.swapaxes(np.tile(ts_filtered.data,[31,180,360,1]),2,3),1,2),0,1)
			#cube.data = high_pass_filter(cube.data,upper_limit_years)
			#cube.data = low_pass_filter(cube.data,lower_limit_years)
			data[model] = {}
			#data[model]['tas'] = {}
			#data[model]['tas'] = istats.pearsonr(ts_filtered_2D,cube,corr_coords=['time'])
			cube = 0
			cube1.data = high_pass_filter(cube1.data,upper_limit_years)
			cube1.data = low_pass_filter(cube1.data,lower_limit_years)
			data[model]['thetao'] = {}
			data[model]['thetao'] = istats.pearsonr(ts_filtered_2D,cube1,corr_coords=['time'])


#failed on MPI-ESM-MR


with open('/home/ph290/Documents/python_scripts/pickles/hiatus.pickle', 'w') as f:
    pickle.dump([data], f)

'''

'''

with open('/home/ph290/Documents/python_scripts/pickles/hiatus.pickle', 'r') as f:
    [data] = pickle.load(f)


models2 = list(data.viewkeys())
#models2.remove('HadGEM2-ES')
no_models = np.size(models2)

thetao_mean = data[models2[0]]['thetao'].copy()
thetao_stdev = thetao_mean.copy()

data_tmp1 = np.empty([no_models,np.shape(data[models2[0]]['thetao'].data)[0],np.shape(data[models2[0]]['thetao'].data)[1],np.shape(data[models2[0]]['thetao'].data)[2]])*0.0+np.NAN



for i,model in enumerate(models2):
    data_tmp1[i,:,:] = data[model]['thetao'].data



thetao_mean.data = np.mean(data_tmp1,axis = 0)
thetao_stdev.data = np.std(data_tmp1,axis = 0)



thetao_mean.coord('latitude').guess_bounds()
thetao_mean.coord('longitude').guess_bounds()

'''

grid_areas = iris.analysis.cartography.area_weights(thetao_mean)


levels = np.linspace(-0.35,0.35,20)

brewer_cmap = mpl_cm.get_cmap('brewer_RdBu_11')
plt.close('all')
#plt.figure(figsize=(12, 12))
fig, axes  = plt.subplots(nrows=4, ncols=3)
plt.subplot(4 ,3, 1)
cs1 = iplt.contourf(thetao_mean.collapsed('depth',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('T (ens. mean)')

plt.subplot(4 ,3, 2)
cs2 = iplt.contourf(thetao_mean.collapsed('longitude',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
#plt.gca().coastlines()
plt.xlabel('lat N. (deg)')
plt.ylabel('depth (m)')
plt.title('Global (ens. mean)')

plt.subplot(4 ,3, 3)
cs3 = iplt.contourf(thetao_mean.collapsed('longitude',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
plt.ylim(350,0)
#plt.gca().coastlines()
plt.xlabel('lat N. (deg)')
plt.ylabel('depth (m)')
plt.title('Global (ens. mean)')


lon_west = -80.0
lon_east = 20.5
lat_south = -90
lat_north =  90

thetao_mean_region = thetao_mean.intersection(longitude=(lon_west, lon_east))
thetao_mean_region = thetao_mean_region.intersection(latitude=(lat_south, lat_north))

grid_areas = iris.analysis.cartography.area_weights(thetao_mean_region)

plt.subplot(4 ,3, 4)
cs4 = iplt.contourf(thetao_mean_region.collapsed('depth',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('Atlantic T (ens. mean)')

plt.subplot(4 ,3, 5)
cs5 = iplt.contourf(thetao_mean_region.collapsed('longitude',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
#plt.gca().coastlines()
plt.xlabel('lat N. (deg)')
plt.ylabel('depth (m)')
plt.title('Atlantic (ens. mean)')

plt.subplot(4 ,3, 6)
cs6 = iplt.contourf(thetao_mean_region.collapsed('longitude',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
plt.ylim(350,0)
#plt.gca().coastlines()
plt.xlabel('lat N. (deg)')
plt.ylabel('depth (m)')
plt.title('Atlantic (ens. mean)')

lon_west = -230
lon_east = -100
lat_south = -90
lat_north =  90

thetao_mean_region = thetao_mean.intersection(longitude=(lon_west, lon_east))
thetao_mean_region = thetao_mean_region.intersection(latitude=(lat_south, lat_north))

grid_areas = iris.analysis.cartography.area_weights(thetao_mean_region)

plt.subplot(4 ,3, 7)
cs7 = iplt.contourf(thetao_mean_region.collapsed('depth',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('Pacific T (ens. mean)')

plt.subplot(4 ,3, 8)
cs8 = iplt.contourf(thetao_mean_region.collapsed('longitude',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
#plt.gca().coastlines()
plt.xlabel('lat N. (deg)')
plt.ylabel('depth (m)')
plt.title('Pacific (ens. mean)')

plt.subplot(4 ,3, 9)
cs9 = iplt.contourf(thetao_mean_region.collapsed('longitude',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
plt.ylim(350,0)
#plt.gca().coastlines()
plt.xlabel('lat N. (deg)')
plt.ylabel('depth (m)')
plt.title('Pacific (ens. mean)')

lon_west = -330
lon_east = -240
lat_south = -90
lat_north =  90

thetao_mean_region = thetao_mean.intersection(longitude=(lon_west, lon_east))
thetao_mean_region = thetao_mean_region.intersection(latitude=(lat_south, lat_north))
grid_areas = iris.analysis.cartography.area_weights(thetao_mean_region)

plt.subplot(4 ,3, 10)
cs10 = iplt.contourf(thetao_mean_region.collapsed('depth',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
plt.gca().coastlines()
plt.title('Indian T (ens. mean)')

plt.subplot(4 ,3, 11)
cs11 = iplt.contourf(thetao_mean_region.collapsed('longitude',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
#plt.gca().coastlines()
plt.xlabel('lat N. (deg)')
plt.ylabel('depth (m)')
plt.title('Indian (ens. mean)')

plt.subplot(4 ,3, 12)
cs12 = iplt.contourf(thetao_mean_region.collapsed('longitude',iris.analysis.MEAN, weights=grid_areas),levels, cmap=brewer_cmap)
plt.ylim(350,0)
plt.xlabel('lat N. (deg)')
plt.ylabel('depth (m)')
#plt.gca().coastlines()
plt.title('Indian (ens. mean)')



fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.1, 0.04, 0.8, 0.02])
cbar = plt.colorbar(cs1,orientation='horizontal', cax=cbar_ax)
plt.tight_layout(pad=2, w_pad=0.2, h_pad=0.2)
plt.savefig('/home/ph290/Documents/figures/hiatus_correlations_depth.pdf')
#plt.show(block = False)


'''

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

'''
