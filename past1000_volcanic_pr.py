'''

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

def dec_to_date(decimal_date):
    year = int(decimal_date)
    rem = decimal_date - year
    base = datetime(year, 1, 1)
    return base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)

def digital(cube):
    cube_out = cube.copy()
    cube_out[np.where(cube >= 0.0)] = 1.0
    cube_out[np.where(cube < 0.0)] = -1.0
    return cube_out

#this is a simple function that we call later to look at the file names and extarct from them a unique list of models to process
#note that the model name is in the filename when downlaode ddirectly from the CMIP5 archive
def model_names(directory,variable,experiment):
	files = glob.glob(directory+'/*'+variable+'_'+experiment+'*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models


#first process: /data/temp/ph290/last_1000

input_directory = '/media/usb_external1/cmip5/reynolds_data/past1000/' # tas

variables = np.array(['pr'])
experiments = ['past1000']
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.

'''
#Main bit of code follows...
'''

experiment = experiments[0]

models = model_names(input_directory,variables[0],experiment)

models = list(models)
#models.remove('bcc-csm1-1')

cube1 = iris.load_cube(input_directory+models[0]+'*'+variables[0]+'*.nc')[0]

models2 = []
cubes0 = []

for model in models:
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory+model+'*'+variables[0]+'_'+experiment+'*.nc')
	except:
		cube = iris.load(input_directory+model+'*'+variables[0]+'_'+experiment+'*.nc')
		cube = cube[0]
	cubes0.append(cube)


coord = cubes[0].coord('time')
dt = coord.units.num2date(coord.points)
model_years = np.array([coord.units.num2date(value).year for value in coord.points])

volcanic_smoothing = 2 #yrs

#########################
#MCA
#########################


min_yr = 850
max_yr = 1250
loc0 = np.where((model_years >= min_yr) & (model_years < max_yr))[0]

cubes = np.copy(cubes0)
for i,cube in enumerate(cubes0):
	cubes[i] = cube[loc0]


'''
#Volcanic forcing
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
data_final = data1.copy()
data_final[:,1] = data



volc_years = data_final[:,0]

loc = np.where((volc_years >= min_yr) & (volc_years < max_yr))[0]

volc_tmp = running_mean_post.running_mean_post(data_final[loc,1],36.0*volcanic_smoothing)


volc_years = data_final[np.arange(np.size(volc_tmp)),0]
yrs = np.floor(data_final[np.arange(np.size(volc_tmp)),0])
yrs_unique = np.unique(yrs)
data_ann = np.empty([np.size(yrs_unique),2])

for i,y in enumerate(yrs_unique):
        loc = np.where(yrs == y)[0]
        data_ann[i,0] = y
        data_ann[i,1] = np.mean(volc_tmp[loc])

volc = data_ann[:,1]

volc_mean = np.mean(volc)
volc_mean = 0.02
high_volc = np.where(volc > volc_mean)[0]
low_volc = np.where(volc < volc_mean)[0]

high_volc_data = np.empty([np.size(cubes),np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
low_volc_data = high_volc_data.copy()
change_volc_data = high_volc_data.copy()
digital_volc_data = high_volc_data.copy()
digital_high = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
digital_low = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])

for i,cube in enumerate(cubes):
	tmp = cube[high_volc].collapsed('time',iris.analysis.MEAN)
	high_volc_data[i,:,:] = tmp.data
	tmp2 = cube[low_volc].collapsed('time',iris.analysis.MEAN)
	low_volc_data[i,:,:] = tmp2.data
	change_volc_data[i,:,:] = tmp.data - tmp2.data
	digital_volc_data[i,:,:] = digital(change_volc_data[i,:,:])

digital_volc_data_tmp1 = digital_volc_data.copy()
digital_volc_data_tmp2 = digital_volc_data.copy()
digital_volc_data_tmp1[np.where(digital_volc_data == 1.0)] = 0
digital_low = np.sum(digital_volc_data_tmp1,axis = 0)
digital_volc_data_tmp2[np.where(digital_volc_data == -1.0)] = 0
digital_high = np.sum(digital_volc_data_tmp2,axis = 0)

high_volc_data_mean = np.mean(high_volc_data,axis = 0)
low_volc_data_mean = np.mean(low_volc_data,axis = 0)
change_volc_data_mean = np.mean(change_volc_data,axis = 0)

high_volc_data_mean_cube = cubes[0][0].copy()
high_volc_data_mean_cube.data = high_volc_data_mean
low_volc_data_mean_cube = cubes[0][0].copy()
low_volc_data_mean_cube.data = low_volc_data_mean
change_volc_data_mean_cube = cubes[0][0].copy()
change_volc_data_mean_cube.data = change_volc_data_mean
digital_low_cube = cubes[0][0].copy()
digital_low_cube.data = digital_low
digital_high_cube = cubes[0][0].copy()
digital_high_cube.data = digital_high

change_digital = change_volc_data_mean_cube.copy()
change_digital.data = digital(change_volc_data_mean_cube.data)

digital_low_cube.data[np.where(change_digital.data == 1.0)] = 0
digital_high_cube.data[np.where(change_digital.data == -1.0)] = 0

digital_cube_final = digital_low_cube.copy()
tmp = np.zeros([digital_cube_final.shape[0],digital_cube_final.shape[1]])
tmp[:] = np.nan
digital_cube_final.data = tmp

#!!!!!!!!!!!#
no_agree = 3
#!!!!!!!!!!!#
digital_cube_final.data[np.where(digital_low_cube.data <= -1.0*no_agree)] = 1
digital_cube_final.data[np.where(digital_high_cube.data >= no_agree)] = 1


#dummy cube
latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
                    units='degrees')
longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
                     units='degrees')
dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')



cube_mca = change_volc_data_mean_cube




#########################
#LIA
#########################

min_yr = 1300
max_yr = 1800
loc0 = np.where((model_years >= min_yr) & (model_years < max_yr))[0]

cubes = np.copy(cubes0)
for i,cube in enumerate(cubes0):
	cubes[i] = cube[loc0]


'''
#Volcanic forcing
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
data_final = data1.copy()
data_final[:,1] = data

volc_years = data_final[:,0]

loc = np.where((volc_years >= min_yr) & (volc_years < max_yr))[0]

volc_tmp = running_mean_post.running_mean_post(data_final[loc,1],36.0*volcanic_smoothing)


volc_years = data_final[np.arange(np.size(volc_tmp)),0]
yrs = np.floor(data_final[np.arange(np.size(volc_tmp)),0])
yrs_unique = np.unique(yrs)
data_ann = np.empty([np.size(yrs_unique),2])

for i,y in enumerate(yrs_unique):
        loc = np.where(yrs == y)[0]
        data_ann[i,0] = y
        data_ann[i,1] = np.mean(volc_tmp[loc])

volc = data_ann[:,1]

volc_mean = np.mean(volc)
volc_mean = 0.02
high_volc = np.where(volc > volc_mean)[0]
low_volc = np.where(volc < volc_mean)[0]

high_volc_data = np.empty([np.size(cubes),np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
low_volc_data = high_volc_data.copy()
change_volc_data = high_volc_data.copy()
digital_volc_data = high_volc_data.copy()
digital_high = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])
digital_low = np.empty([np.shape(cubes[0][0].data)[0],np.shape(cubes[0][0].data)[1]])

for i,cube in enumerate(cubes):
	tmp = cube[high_volc].collapsed('time',iris.analysis.MEAN)
	high_volc_data[i,:,:] = tmp.data
	tmp2 = cube[low_volc].collapsed('time',iris.analysis.MEAN)
	low_volc_data[i,:,:] = tmp2.data
	change_volc_data[i,:,:] = tmp.data - tmp2.data
	digital_volc_data[i,:,:] = digital(change_volc_data[i,:,:])

digital_volc_data_tmp1 = digital_volc_data.copy()
digital_volc_data_tmp2 = digital_volc_data.copy()
digital_volc_data_tmp1[np.where(digital_volc_data == 1.0)] = 0
digital_low = np.sum(digital_volc_data_tmp1,axis = 0)
digital_volc_data_tmp2[np.where(digital_volc_data == -1.0)] = 0
digital_high = np.sum(digital_volc_data_tmp2,axis = 0)

high_volc_data_mean = np.mean(high_volc_data,axis = 0)
low_volc_data_mean = np.mean(low_volc_data,axis = 0)
change_volc_data_mean = np.mean(change_volc_data,axis = 0)

high_volc_data_mean_cube = cubes[0][0].copy()
high_volc_data_mean_cube.data = high_volc_data_mean
low_volc_data_mean_cube = cubes[0][0].copy()
low_volc_data_mean_cube.data = low_volc_data_mean
change_volc_data_mean_cube = cubes[0][0].copy()
change_volc_data_mean_cube.data = change_volc_data_mean
digital_low_cube = cubes[0][0].copy()
digital_low_cube.data = digital_low
digital_high_cube = cubes[0][0].copy()
digital_high_cube.data = digital_high

change_digital = change_volc_data_mean_cube.copy()
change_digital.data = digital(change_volc_data_mean_cube.data)

digital_low_cube.data[np.where(change_digital.data == 1.0)] = 0
digital_high_cube.data[np.where(change_digital.data == -1.0)] = 0

digital_cube_final = digital_low_cube.copy()
tmp = np.zeros([digital_cube_final.shape[0],digital_cube_final.shape[1]])
tmp[:] = np.nan
digital_cube_final.data = tmp

#!!!!!!!!!!!#
no_agree = 3
#!!!!!!!!!!!#
digital_cube_final.data[np.where(digital_low_cube.data <= -1.0*no_agree)] = 1
digital_cube_final.data[np.where(digital_high_cube.data >= no_agree)] = 1


#dummy cube
latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
                    units='degrees')
longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
                     units='degrees')
dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')



cube_lia = change_volc_data_mean_cube

'''

minz = -0.000001
maxz = 0.000001
new_cube_lia = cube_lia.copy()
new_cube_lia.data[np.where(new_cube_lia.data < minz)] = minz
new_cube_lia.data[np.where(new_cube_lia.data > maxz)] = maxz


new_cube = cube_mca.copy()
new_cube = cube_lia-cube_mca
# - cube_mca
new_cube.data[np.where(new_cube.data < minz)] = minz
new_cube.data[np.where(new_cube.data > maxz)] = maxz

plt.close('all')

fig = plt.figure()
ax2 = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax2.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
my_plot1 = iplt.contourf(new_cube,np.linspace(minz,maxz,31),cmap='bwr')
plt.gca().coastlines()

bar = plt.colorbar(my_plot1, orientation='horizontal', extend='both')
bar.set_label(new_cube_lia.long_name+' ('+format(new_cube_lia.units)+')')

#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/lia_minus_mca_p.png')


