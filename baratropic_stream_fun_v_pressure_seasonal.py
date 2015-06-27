'''

import cartopy.crs as ccrs
import iris.plot as iplt
import cartopy.feature as cfeature
import iris
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import glob
import os
import iris.quickplot as qplt
import scipy


def model_names_psl(directory):
        files = glob.glob(directory+'/*psl*.nc')
        models_tmp = []
        for file in files:
                statinfo = os.stat(file)
                if statinfo.st_size >= 1:
                        models_tmp.append(file.split('/')[-1].split('_')[0])
                        models = np.unique(models_tmp)
        return models

def model_names_msftbarot(directory):
        files = glob.glob(directory+'/*msftbarot*.nc')
        models_tmp = []
        for file in files:
                statinfo = os.stat(file)
                if statinfo.st_size >= 1:
                        models_tmp.append(file.split('/')[-1].split('_')[0])
                        models = np.unique(models_tmp)
        return models

N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
################
middle_cuttoff_high=20.0
################

Wn_mid_high=timestep_between_values/middle_cuttoff_high

b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')


directory = '/media/usb_external1/cmip5/piControl_summer/'

models_I = model_names_psl(directory)
models_II = model_names_msftbarot(directory)

models = list(set(models_I).intersection(models_II))

data = {}

for model in models:
        print model
	cube1 = iris.load_cube(directory+model+'_msftbarot*.nc')

	n = 65.0
	s = 35.0
	e = -10.0
	w = -60.0

	cube_region_tmp = cube1.intersection(longitude=(w, e))
	cube_region = cube_region_tmp.intersection(latitude=(s, n))

	spg_strength = np.min(cube_region.data,axis = 1)
	spg_strength = np.min(spg_strength,axis = 1)

	coord = cube1.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])
	data[model] = {}
	data[model]['year'] = year
	#data[model]['msftbarot'] = signal.detrend(spg_strength)
	data[model]['msftbarot'] = scipy.signal.filtfilt(b2, a2, spg_strength)

maps = {}

for model in models:
	print model
	data_minus_mean = data[model]['msftbarot'] - np.mean(data[model]['msftbarot'])
	data_stdev = np.std(data_minus_mean)
	loc = np.where(data_minus_mean > data_stdev)
	high_years = data[model]['year'][loc]
	loc = np.where(data_minus_mean < data_stdev*-1.0)
	low_years = data[model]['year'][loc]

	cube2 = iris.load_cube(directory+model+'_psl*.nc')
	coord = cube2.coord('time')
	dt = coord.units.num2date(coord.points)
	year = np.array([coord.units.num2date(value).year for value in coord.points])	

	high_indicies = np.nonzero(np.in1d(high_years, year))[0]	
	low_indicies = np.nonzero(np.in1d(low_years, year))[0]
	
	high_indicies_p1 = np.nonzero(np.in1d(high_years+2, year))[0]	
	low_indicies_p1 = np.nonzero(np.in1d(low_years+2, year))[0]
	high_indicies_p2 = np.nonzero(np.in1d(high_years+4, year))[0]	
	low_indicies_p2 = np.nonzero(np.in1d(low_years+4, year))[0]
	high_indicies_p3 = np.nonzero(np.in1d(high_years+6, year))[0]	
	low_indicies_p3 = np.nonzero(np.in1d(low_years+6, year))[0]

	high_indicies_m1 = np.nonzero(np.in1d(high_years-2, year))[0]	
	low_indicies_m1 = np.nonzero(np.in1d(low_years-2, year))[0]
	high_indicies_m2 = np.nonzero(np.in1d(high_years-4, year))[0]	
	low_indicies_m2 = np.nonzero(np.in1d(low_years-4, year))[0]
	high_indicies_m3 = np.nonzero(np.in1d(high_years-6, year))[0]	
	low_indicies_m3 = np.nonzero(np.in1d(low_years-6, year))[0]

	high_psl = cube2[high_indicies].collapsed('time',iris.analysis.MEAN)
	low_psl = cube2[low_indicies].collapsed('time',iris.analysis.MEAN)

	high_psl_minus_1 = cube2[high_indicies_m1].collapsed('time',iris.analysis.MEAN)
	low_psl_minus_1 = cube2[low_indicies_m1].collapsed('time',iris.analysis.MEAN)
	high_psl_minus_2 = cube2[high_indicies_m2].collapsed('time',iris.analysis.MEAN)
	low_psl_minus_2 = cube2[low_indicies_m2].collapsed('time',iris.analysis.MEAN)
	high_psl_minus_3 = cube2[high_indicies_m3].collapsed('time',iris.analysis.MEAN)
	low_psl_minus_3 = cube2[low_indicies_m3].collapsed('time',iris.analysis.MEAN)

	high_psl_plus_1 = cube2[high_indicies_p1].collapsed('time',iris.analysis.MEAN)
	low_psl_plus_1 = cube2[low_indicies_p1].collapsed('time',iris.analysis.MEAN)
	high_psl_plus_2 = cube2[high_indicies_p2].collapsed('time',iris.analysis.MEAN)
	low_psl_plus_2 = cube2[low_indicies_p2].collapsed('time',iris.analysis.MEAN)
	high_psl_plus_3 = cube2[high_indicies_p3].collapsed('time',iris.analysis.MEAN)
	low_psl_plus_3 = cube2[low_indicies_p3].collapsed('time',iris.analysis.MEAN)
	
	maps[model] = {}
	maps[model]['high'] = high_psl
	maps[model]['low'] = low_psl
	maps[model]['difference'] = high_psl - low_psl
	maps[model]['difference_plus_1'] = high_psl_plus_1 - low_psl_plus_1
	maps[model]['difference_plus_2'] = high_psl_plus_2 - low_psl_plus_2
	maps[model]['difference_plus_3'] = high_psl_plus_3 - low_psl_plus_3
	maps[model]['difference_minus_1'] = high_psl_minus_1 - low_psl_minus_1
	maps[model]['difference_minus_2'] = high_psl_minus_2 - low_psl_minus_2
	maps[model]['difference_minus_3'] = high_psl_minus_3 - low_psl_minus_3


'''
	
shape = np.shape(cube2)


all_data_diff = np.zeros([np.size(models),shape[1],shape[2]])
all_data_diff[:] = np.NAN
all_data_low = all_data_diff.copy()
all_data_high = all_data_diff.copy()
all_data_diff_plus_1 = all_data_diff.copy()
all_data_diff_plus_2 = all_data_diff.copy()
all_data_diff_plus_3 = all_data_diff.copy()
all_data_diff_minus_1 = all_data_diff.copy()
all_data_diff_minus_2 = all_data_diff.copy()
all_data_diff_minus_3 = all_data_diff.copy()


for i,model in enumerate(models):
	all_data_low[i,:,:] = maps[model]['low'].data
	all_data_high[i,:,:] = maps[model]['high'].data
	all_data_diff[i,:,:] = maps[model]['difference'].data
	all_data_diff_plus_1[i,:,:] = maps[model]['difference_plus_1'].data
	all_data_diff_plus_2[i,:,:] = maps[model]['difference_plus_2'].data
	all_data_diff_plus_3[i,:,:] = maps[model]['difference_plus_3'].data
	all_data_diff_minus_1[i,:,:] = maps[model]['difference_minus_1'].data
	all_data_diff_minus_2[i,:,:] = maps[model]['difference_minus_2'].data
	all_data_diff_minus_3[i,:,:] = maps[model]['difference_minus_3'].data


all_data_diff_mean = np.mean(all_data_diff,axis = 0)
all_data_high_mean = np.mean(all_data_high,axis = 0)
all_data_low_mean = np.mean(all_data_low,axis = 0)
all_data_diff_mean_plus_1 = np.mean(all_data_diff_plus_1,axis = 0)
all_data_diff_mean_plus_2 = np.mean(all_data_diff_plus_2,axis = 0)
all_data_diff_mean_plus_3 = np.mean(all_data_diff_plus_3,axis = 0)
all_data_diff_mean_minus_1 = np.mean(all_data_diff_minus_1,axis = 0)
all_data_diff_mean_minus_2 = np.mean(all_data_diff_minus_2,axis = 0)
all_data_diff_mean_minus_3 = np.mean(all_data_diff_minus_3,axis = 0)

all_data_diff_mean_cube = cube2[0].copy()
all_data_high_mean_cube = cube2[0].copy()
all_data_low_mean_cube = cube2[0].copy()
all_data_diff_mean_cube_plus_1 = cube2[0].copy()
all_data_diff_mean_cube_plus_2 = cube2[0].copy()
all_data_diff_mean_cube_plus_3 = cube2[0].copy()
all_data_diff_mean_cube_minus_1 = cube2[0].copy()
all_data_diff_mean_cube_minus_2 = cube2[0].copy()
all_data_diff_mean_cube_minus_3 = cube2[0].copy()

all_data_diff_mean_cube.data = all_data_diff_mean
all_data_high_mean_cube.data = all_data_high_mean
all_data_low_mean_cube.data = all_data_low_mean
all_data_diff_mean_cube_plus_1.data = all_data_diff_mean_plus_1
all_data_diff_mean_cube_plus_2.data = all_data_diff_mean_plus_2
all_data_diff_mean_cube_plus_3.data = all_data_diff_mean_plus_3
all_data_diff_mean_cube_minus_1.data = all_data_diff_mean_minus_1
all_data_diff_mean_cube_minus_2.data = all_data_diff_mean_minus_2
all_data_diff_mean_cube_minus_3.data = all_data_diff_mean_minus_3

# plt.close('all')
# fig = plt.figure(figsize=([15,5]))
# ax1 = fig.add_subplot(131)
# qplt.contourf(all_data_high_mean_cube,31)
# plt.gca().coastlines()
# ax1 = fig.add_subplot(132)
# qplt.contourf(all_data_low_mean_cube,31)
# plt.gca().coastlines()
# ax1 = fig.add_subplot(133)
# qplt.contourf(all_data_diff_mean_cube,31)
# plt.gca().coastlines()
# plt.show(block = False)



zmin = 99000.0
zmax = 102800.0

zmin2 = -4
zmax2 = +4


plt.close('all')
fig = plt.figure(figsize = (20,10))
ax = plt.subplot(131,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_high_mean_cube,np.linspace(zmin,zmax,31),cmap='Spectral')
plt.gca().coastlines()
bar = plt.colorbar(my_plot,ticks=[zmin, 0, zmax], orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP Strong Subpolar Gyre')

ax1 = plt.subplot(132,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_low_mean_cube,np.linspace(zmin,zmax,31),cmap='Spectral')
plt.gca().coastlines()
bar = plt.colorbar(my_plot,ticks=[zmin, 0, zmax], orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP Weak Subpolar Gyre')

ax1 = plt.subplot(133,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_diff_mean_cube,np.linspace(zmin2,zmax2,31),cmap='bwr')
plt.gca().coastlines()
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP High minus Low Subpolar Gyre')

#plt.show(block = True)
plt.savefig('/home/ph290/Documents/figures/spg_mslp_1_summer.png')



zmin = -6
zmax = +6


plt.close('all')
fig = plt.figure(figsize = (20,20))
ax = plt.subplot(231,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_diff_mean_cube_minus_3,np.linspace(zmin,zmax,31),cmap='bwr')
plt.gca().coastlines()
bar = plt.colorbar(my_plot,ticks=[zmin, 0, zmax], orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP -6')

ax1 = plt.subplot(232,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_diff_mean_cube_minus_2,np.linspace(zmin,zmax,31),cmap='bwr')
plt.gca().coastlines()
bar = plt.colorbar(my_plot,ticks=[zmin, 0, zmax], orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP -4')

ax1 = plt.subplot(233,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_diff_mean_cube_minus_1,np.linspace(zmin,zmax,31),cmap='bwr')
plt.gca().coastlines()
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP -2')

ax = plt.subplot(234,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_diff_mean_cube_plus_1,np.linspace(zmin,zmax,31),cmap='bwr')
plt.gca().coastlines()
bar = plt.colorbar(my_plot,ticks=[zmin, 0, zmax], orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP +2')

ax1 = plt.subplot(235,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_diff_mean_cube_plus_2,np.linspace(zmin,zmax,31),cmap='bwr')
plt.gca().coastlines()
bar = plt.colorbar(my_plot,ticks=[zmin, 0, zmax], orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP +4')

ax1 = plt.subplot(236,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot = iplt.contourf(all_data_diff_mean_cube_plus_3,np.linspace(zmin,zmax,31),cmap='bwr')
plt.gca().coastlines()
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(all_data_high_mean_cube.long_name+' ('+format(all_data_high_mean_cube.units)+')')
plt.title('Summer MSLP +6')


#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/spg_mslp_2_summer.png')


#offsets


'''

file = '/home/ph290/data0/misc_data/gnipcl-2.csv'

data = np.genfromtxt(file,delimiter =',',skip_header = 1,usecols=[2,3,7])

lats = np.round(data[:,0])
lons = np.round(data[:,1])

d18O = all_data_diff_mean_cube.copy()
d18O.data[:] = np.NAN

tmp_data = np.zeros([180,360])
tmp_data[:] = np.NAN

for i,la in enumerate(all_data_diff_mean_cube.coord('latitude').points+0.5):
	for j,lo in enumerate(all_data_diff_mean_cube.coord('longitude').points-180.0):
		loc = np.where((lats == la) & (lons == lo))[0]		
		tmp_data[i,j] = np.mean(data[loc,2])


d18O.data = tmp_data
d18O.data = np.ma.masked_invalid(d18O.data)

qplt.contourf(d18O,np.linspace(-40,10,31))
plt.show()

'''

