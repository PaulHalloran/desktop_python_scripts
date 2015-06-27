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
import running_mean
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import running_mean
import cartopy.crs as ccrs
import iris.analysis.cartography
import numpy.ma as ma
import scipy.interpolate
import gc

#execfile('cmip5_stm_func_II.py')

'''
#producing an Atlantic mask (mask masked and Atlantic has value of 1, elsewhere zero)
'''

input_file = '/media/usb_external1/cmip5/last1000/CCSM4_vo_past1000_regridded.nc'
cube = iris.load_cube(input_file)

tmp_cube = cube[0,16].copy()
tmp_cube = tmp_cube*0.0

location = -30

for y in np.arange(180):
    flag = 0
    tmp = tmp_cube.data.mask[y,:]
    tmp2 = tmp_cube.data[y,:]
    for x in np.arange(360):
        if tmp[location] == True:
            flag = 1
        if ((tmp[location] == False) & (flag == 0)):
            tmp2[location] = 1
        tmp = np.roll(tmp,+1)
        tmp2 = np.roll(tmp2,+1)
    tmp_cube.data.mask[y,:] = tmp
    tmp_cube.data.data[y,:] = tmp2.data

location = location+1

for y in np.arange(180):
    flag = 0
    tmp = tmp_cube.data.mask[y,:]
    tmp2 = tmp_cube.data[y,:]
    for x in np.arange(360):
        if tmp[location] == True:
            flag = 1
        if ((tmp[location] == False) & (flag == 0)):
            tmp2[location] = 1
        tmp = np.roll(tmp,-1)
        tmp2 = np.roll(tmp2,-1)
    tmp_cube.data.mask[y,:] = tmp
    tmp_cube.data.data[y,:] = tmp2.data

tmp_cube.data.data[150:180,:] = 0.0
tmp_cube.data.data[0:40,:] = 0.0
tmp_cube.data.data[:,20:180] = 0.0
tmp_cube.data.data[:,180:280] = 0.0

loc = np.where(tmp_cube.data.data == 0.0)
tmp_cube.data.mask[loc] = True

mask1 = tmp_cube.data.mask


# plt.close('all')
# qplt.contourf(cube[0,28])
# plt.show(block = False)

# my_dir = '/media/usb_external1/cmip5/msftmyz/last1000/'
# files = np.array(glob.glob(my_dir+'/*'+'MPI-ESM-P'+'_*msftmy*.nc'))
# cube2 = iris.load_cube(files)[:,0,:,:]
# loc = np.where((cube2.coord('grid_latitude').points >= 30) & (cube2.coord('grid_latitude').points <= 50))
# lat = cube2.coord('grid_latitude').points[loc]
# sub_cube = cube2.extract(iris.Constraint(grid_latitude = lat))
# stream_function_tmp = sub_cube.collapsed(['depth','grid_latitude'],iris.analysis.MAX)
# coord = stream_function_tmp.coord('time')
# dt = coord.units.num2date(coord.points)
# year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
# tmp = stream_function_tmp.data/1.0e9
# mpi_official_strmfun_26 = (tmp[np.logical_not(np.isnan(tmp))])
# mpi_official_year = (year_tmp[np.logical_not(np.isnan(tmp))])

# f0 = plt.figure()
# qplt.contourf(cube2[0],51)
# plt.show(block = False)


'''
#calculating stream function
'''

files = glob.glob('/media/usb_external1/cmip5/last1000_vo_amoc/*_vo_*.nc')

models = []
max_strm_fun = []
model_years = []

for file in files:

model = file.split('/')[5].split('_')[0]
print model
models.append(model)
cube = iris.load_cube(file)

print 'applying mask'

try:
        levels =  np.arange(cube.coord('depth').points.size)
except:
        levels = np.arange(cube.coord('ocean sigma over z coordinate').points.size)

for level in levels:
        print 'level: '+str(level)
        for year in np.arange(cube.coord('time').points.size):
                #print 'year: '+str(year)
            mask2 = cube.data.mask[year,level,:,:]
            tmp_mask = np.ma.mask_or(mask1, mask2)
            cube.data.mask[year,level,:,:] = tmp_mask
            gc.collect()


cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(cube[0])
grid_areas = np.sqrt(grid_areas)

shape = np.shape(cube)
tmp = cube[0].collapsed('longitude',iris.analysis.SUM)
collapsed_data = np.tile(tmp.data,[shape[0],1,1])

print 'collapsing cube along longitude'
try:
        slices = cube.slices(['depth', 'latitude','longitude'])
except:
        slices = cube.slices(['ocean sigma over z coordinate', 'latitude','longitude'])
for i,t_slice in enumerate(slices):
        print 'year:'+str(i)
        t_slice *= grid_areas
        collapsed_data[i] = t_slice.collapsed('longitude',iris.analysis.SUM).data

try:
        depths = cube.coord('depth').points*-1.0
        bounds = cube.coord('depth').bounds
except:
        depths = cube.coord('ocean sigma over z coordinate').points*-1.0
        bounds = cube.coord('ocean sigma over z coordinate').bounds	
thickness = bounds[:,1] - bounds[:,0]
test = thickness.mean()
if test > 1:
        thickness = bounds[1:,0] - bounds[0:-1,0]
        thickness = np.append(thickness, thickness[-1])

thickness = np.flipud(np.rot90(np.tile(thickness,[180,1])))

tmp_strm_fun = []
for i in np.arange(np.size(collapsed_data[:,0,0])):
        tmp = collapsed_data[i].copy()
        tmp = tmp*thickness
        tmp = np.cumsum(tmp,axis = 1)
        tmp = tmp*-1.0*1.0e-3
        tmp *= 1029.0 #conversion from m3 to kg
        #tmp = tmp*1.0e-7*0.8 # no idea why I need to do this conversion - check...
        coord = t_slice.coord('latitude').points
        loc = np.where(coord >= 45)[0][0]
        tmp_strm_fun = np.append(tmp_strm_fun,np.max(tmp[:,loc]))

coord = cube.coord('time')
dt = coord.units.num2date(coord.points)
years = np.array([coord.units.num2date(value).year for value in coord.points])
model_years.append(years)

max_strm_fun.append(tmp_strm_fun)


# plt.close('all')
# # plt.plot(tmp_strm_fun/(110*110))
# plt.plot(tmp_strm_fun)
# plt.plot(mpi_official_strmfun_26)
# plt.show(block = False)

# f2 = plt.figure()		
# CS = plt.contourf(np.arange(180)-90,depths,collapsed_data[0],51)
# plt.colorbar(CS)
# plt.show(block = False)

# f3 = plt.figure()
# CS = plt.contourf(np.arange(180)-90,depths,np.fliplr(tmp*-1.0*1.0e-3),51)
# plt.colorbar(CS)
# plt.show(block = False)


for i,model in enumerate(models):
	plt.plot(model_years[i],max_strm_fun[i])


plt.show(block = False)

###
#read in temperature
###



amo_box_tas = []
model_years_tas = []

for i,model in enumerate(models):
	print 'processing: '+model
	file = glob.glob('/media/usb_external1/cmip5/tas_regridded/'+model+'_tas_past1000_regridded.nc')
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

linestyles = ['-','--','-.']

smoothing_val = 10

plt.close('all')
fig = plt.figure(figsize = (6,20))

for i,model in enumerate(models):
        ax1 = fig.add_subplot(np.size(models),1,i+1)
        tmp = amo_box_tas[i].data
	tmp = running_mean.running_mean(tmp,smoothing_val)
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years_tas[i] <= 1850))
        tmp = tmp[loc]
        ax1.plot(model_years_tas[i][loc],signal.detrend(tmp),'r',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
        loc = np.where((amo_yr >= 850) & (amo_yr <= 1850))
        ax1.plot(amo_yr[loc],signal.detrend(amo_data[loc]),'k',linewidth=2,alpha = 0.5)
        ax2 = ax1.twinx()
        tmp = max_strm_fun[i]
	tmp = running_mean.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years[i] <= 1850))
        tmp = tmp[loc]
        ax2.plot(model_years[i][loc],signal.detrend(tmp),'b',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
        ax3 = ax2.twinx()
        ax3.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=2,alpha = 0.2)
	#ax3.plot(voln_s[:,0],voln_n[:,1],'b',linewidth=2,alpha = 0.2)
        ax3.set_ylim([0,0.8])
        ax1.set_xlim([800,1850])
	ax1.set_title(model)

plt.tight_layout()
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/amo_fig2.png')

#AVERAGE TOGETHER!

smoothing_val = 10

all_years = np.linspace(850,1850,(1851-850))
average_tas = np.empty([np.size(all_years),np.size(models)-4])
average_tas[:] = np.NAN
average_str = np.empty([np.size(all_years),np.size(models)-4])
average_str[:] = np.NAN

counter = -1
for i,model in enumerate(models):
    if model not in ['FGOALS-gl','GISS-E2-R']:
        counter += 1
        tmp = amo_box_tas[i].data
        tmp = running_mean.running_mean(tmp,smoothing_val)
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years_tas[i] <= 1850) & (model_years_tas[i] >= 850))
        tmp = tmp[loc]
        yrs = model_years_tas[i][loc]
        data = signal.detrend(tmp)
        for j,yr in enumerate(all_years):
                try:
                        loc2 = np.where(yrs == yr)
                        average_tas[j,counter] = data[loc2]
                except:
                        None
        tmp = max_strm_fun[i]
        tmp = running_mean.running_mean(tmp,smoothing_val)
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years[i] <= 1850) & (model_years[i] >= 850))
        tmp = tmp[loc]
        yrs = model_years[i][loc]
        data = signal.detrend(tmp)
        data = data/np.mean(data)
        for j,yr in enumerate(all_years):
                try:
                        loc2 = np.where(yrs == yr)
                        average_str[j,counter] = data[loc2]
                except:
                        None
        

	
average_tas2 = np.mean(average_tas,axis = 1)
average_str2 = np.mean(average_str,axis = 1)

plt.close('all')
fig = plt.figure(figsize = (12,5))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(all_years,average_tas2,'r',linewidth=3,alpha = 0.5,linestyle = linestyles[0])
ax1.plot(amo_yr,signal.detrend(amo_data),'k',linewidth=3,alpha = 0.5)
ax2 = ax1.twinx()
ax2.plot(all_years,average_str2,'b',linewidth=3,alpha = 0.5,linestyle = linestyles[0])
ax3 = ax2.twinx()
ax3.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=3,alpha = 0.2)
#ax3.plot(voln_s[:,0],voln_n[:,1],'b',linewidth=2,alpha = 0.2)
ax3.set_ylim([0,0.8])
ax1.set_xlim([850,1850])
ax1.set_ylim(-0.5,0.5)
ax1.set_title(model)

plt.tight_layout()
plt.show(block = False)
#plt.savefig('/home/ph290/Documents/figures/amo_fig3.png')

