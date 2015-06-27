
import iris.plot as iplt
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris
import glob
import iris.experimental.concatenate
import iris.analysis
import iris.quickplot as qplt
import iris.analysis.cartography
import cartopy.crs as ccrs
import subprocess
from iris.coords import DimCoord
import iris.coord_categorisation
import matplotlib as mpl
import gc
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
import monthly_to_yearly as m2yr
from matplotlib import mlab
import matplotlib.mlab as ml
import cartopy
import running_mean
import matplotlib.cm as mpl_cm
import scipy.stats


def digital(cube):
    cube_out = cube.copy()
    cube_out[np.where(cube >= 0.0)] = 1.0
    cube_out[np.where(cube < 0.0)] = -1.0
    return cube_out

def index_of_array_items_in_another(x,y):
	index = np.argsort(x)
	sorted_x = x[index]
	sorted_index = np.searchsorted(sorted_x, y)
	yindex = np.take(index, sorted_index)
	return np.ma.array(yindex)
	
	
variables = np.array(['tas','pr','rsds'])
z = np.array([0.05,2.5e-7,0.5])
no_agree = np.array([4,5,2])
# 4 or 6
# 5 of 7
# 2 of 3
#note - not enough good models with stream function and rsds to do that that way - will have to do from ssts

my_dir = '/media/usb_external1/cmip5/msftmyz/last1000/'
files1 = np.array(glob.glob(my_dir+'*msftmyz*.nc'))

'''
#which models do we have?
'''

models1 = []
for file in files1:
	models1.append(file.split('/')[-1].split('_')[0])


models_unique = np.unique(np.array(models1))

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



'''
#read in AMOC
'''

smoothing_val = 10

cmip5_max_strmfun = []
cmip5_year = []

cmip5_iceland_tas = []

models_unique = models_unique.tolist()
#models_unique.remove('MRI-CGCM3')

for model in models_unique:
	print model
	files = np.array(glob.glob(my_dir+'/*'+model+'_*msftmy*.nc'))
	cube = iris.load_cube(files)[:,0,:,:]
	try:
		loc = np.where((cube.coord('grid_latitude').points >= 30) & (cube.coord('grid_latitude').points <= 50))
		lat = cube.coord('grid_latitude').points[loc]
		sub_cube = cube.extract(iris.Constraint(grid_latitude = lat))
		stream_function_tmp = sub_cube.collapsed(['depth','grid_latitude'],iris.analysis.MAX)
	except:
		loc = np.where((cube.coord('latitude').points >= 30) & (cube.coord('latitude').points <= 50))
		lat = cube.coord('latitude').points[loc]	
		sub_cube = cube.extract(iris.Constraint(latitude = lat))
		stream_function_tmp = sub_cube.collapsed(['depth','latitude'],iris.analysis.MAX)
	coord = stream_function_tmp.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	#tmp = running_mean.running_mean(signal.detrend(stream_function_tmp.data/1.0e9),1)
	tmp = running_mean.running_mean(stream_function_tmp.data/1.0e9,smoothing_val)
	cmip5_max_strmfun.append(np.ma.masked_invalid(tmp))
	cmip5_year.append(np.ma.masked_invalid(year_tmp))
	
	try:
		file2 = '/media/usb_external1/cmip5/tas_regridded/'+model+'_tas_past1000_regridded.nc'
		cube2 = iris.load_cube(file2)
	except:
		file2 = '/media/usb_external1/cmip5/last1000/'+model+'_thetao_past1000_regridded.nc'
		cube2 = iris.load_cube(file2)
		cube2 = cube2.extract(iris.Constraint(depth = 5))
		
	lon_west = -20.0
	lon_east = -10
	lat_south = 67
	lat_north = 75.0 
	
	lon_west = -75
	lon_east = -7.5
	lat_south = 0
	lat_north = 60.0 

	cube3 = cube2.intersection(longitude=(lon_west, lon_east))
	cube3 = cube3.intersection(latitude=(lat_south, lat_north))

	cube3.coord('latitude').guess_bounds()
	cube3.coord('longitude').guess_bounds()
	grid_areas = iris.analysis.cartography.area_weights(cube3)
	tmp = running_mean.running_mean(cube3.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights = grid_areas).data,smoothing_val)
	cmip5_iceland_tas.append(np.ma.masked_invalid(tmp))


amo_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
amo = np.genfromtxt(amo_file, skip_header = 4)
amo_yr = amo[:,0]
amo_data = amo[:,1]
loc = np.where(amo_yr <= 1850)
amo_yr = amo_yr[loc]
amo_data = amo_data[loc]
amo_data = signal.detrend(amo_data)

plt.close('all')
# fig = plt.figure(figsize = (8,10))
# ax1 = fig.add_subplot(np.size(cmip5_year),1,1)

## y = signal.detrend(np.log(voln_n[:,1]))
#y = np.log10(voln_n[:,1])
##y[np.isnan(y)]=0 
## y[np.isinf(y)]=0
#y = running_mean.running_mean(y,10)
#y2 = np.log(voln_s[:,1])
#y2 = running_mean.running_mean(y2,10)
##ax1.plot(voln_n[:,0],y,'r',linewidth=3,alpha = 0.5)
## ax1.scatter(voln_n[:,0],np.log(voln_n[:,1]),color='r',alpha = 0.03,s=50)
## ax1.scatter(voln_s[:,0],np.log(voln_s[:,1]),color='b',alpha = 0.1)
## ax1.set_ylim([0.0,1.5])
#
#ax1.scatter(voln_n[:,0],y,color='r',alpha = 0.05,s=30)
## ax1.scatter(voln_n[:,0],y2,color='b',alpha = 0.05,s=30)
##ax3 = ax1.twinx()

size = 0
years = 0
for item in cmip5_year:
	if np.size(item) > size:
		years =  item
	size = np.max([size,np.size(item)])
	
mean_years = years

output = np.ma.empty([size,np.size(models_unique)])
output[:,:] = np.NAN
    
for j,dummy in enumerate(models_unique):
	for k,yr in enumerate(cmip5_year[j]):
		loc = np.where(years == yr)
		if np.size(loc[0]) > 0:
			output[loc[0],j] = cmip5_max_strmfun[j][k]

output = np.ma.masked_invalid(output)
multi_mean_strfun = np.ma.mean(output,axis=1)

output = np.ma.empty([size,np.size(models_unique)])
output[:,:] = np.NAN
    
for j,dummy in enumerate(models_unique):
	for k,yr in enumerate(cmip5_year[j]):
		loc = np.where(years == yr)
		if np.size(loc[0]) > 0:
			output[loc[0],j] = cmip5_iceland_tas[j][k]

output = np.ma.masked_invalid(output)
multi_mean_tas = np.ma.mean(output,axis=1)

colors = ['r','g','b']
linestyles = ['-','--','-.']

plt.close('all')
fig = plt.figure(figsize = (8,10))

for i,model in enumerate(models_unique):
	ax1 = fig.add_subplot(np.size(cmip5_year)+1,1,i+1)
	tmp = cmip5_iceland_tas[i]
	loc = np.where(np.logical_not(np.isnan(tmp)))
	tmp = tmp[loc]
	ax1.plot(cmip5_year[i][loc],signal.detrend(tmp),'r',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
	loc = np.where((amo_yr >= 850) & (amo_yr <= 1850))
	ax1.plot(amo_yr[loc],signal.detrend(amo_data[loc]),'k',linewidth=2,alpha = 0.5)
	ax2 = ax1.twinx()
	tmp = cmip5_max_strmfun[i]
	loc = np.where(np.logical_not(np.isnan(tmp)))
	tmp = tmp[loc]
	ax2.plot(cmip5_year[i][loc],signal.detrend(tmp),'b',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
	ax3 = ax2.twinx()
	ax3.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=2,alpha = 0.2)
# 	ax3.plot(voln_s[:,0],voln_n[:,1],'b',linewidth=2,alpha = 0.2)
	ax3.set_ylim([0,0.8])
	ax1.set_xlim([800,1850])
	ax1.set_title(model)

ax1 = fig.add_subplot(np.size(cmip5_year)+1,1,i+2)
tmp = multi_mean_tas
loc = np.where(tmp <> 0)
tmp = tmp[loc]
plt.plot(mean_years[loc],signal.detrend(tmp),'y',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
loc = np.where((amo_yr >= 850) & (amo_yr <= 1850))
ax1.plot(amo_yr[loc],signal.detrend(amo_data[loc]),'k',linewidth=2,alpha = 0.5)
ax2 = ax1.twinx()
tmp = multi_mean_strfun
loc = np.where(tmp <> 0)
tmp = tmp[loc]
ax2.plot(mean_years[loc],signal.detrend(tmp),'y',linewidth=2,alpha = 0.5,linestyle = linestyles[1])


multi_mean_tas
ax1.set_xlim([800,1850])

plt.tight_layout()
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/amo_fig.png')
