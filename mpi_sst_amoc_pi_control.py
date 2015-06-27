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

def my_callback(cube,field, files):
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    cube.attributes.pop('table_id')
    #if np.size(cube) > 1:
    #cube = iris.experimental.concatenate.concatenate(cube)


def extract_data(data_in):
    data_out = data_in.data
    return data_out


def regrid_data_0(file,variable_name,out_filename):
    p = subprocess.Popen("cdo remapbil,r360x180 -selname,"+variable_name+" "+file+" "+out_filename,shell=True)
    p.wait()


directory = '/home/ph290/data0/cmip5_data/'
files = np.array(glob.glob(directory+'msftmyz/piControl/*.nc'))

model_name = 'MPI-ESM-MR'

'''
which models do we have?
'''

models = []
for file in files:
    models.append(file.split('/')[-1].split('_')[2])

models_unique = np.unique(np.array(models))

'''
read in MPI
'''

def remove_anonymous(cube, field, filename):
# this only loads in the region relating to the atlantic
	cube.attributes.pop('creation_date')
	cube.attributes.pop('tracking_id')
	cube = cube[:,0,:,:]
	return cube


files = np.array(glob.glob(directory+'msftmyz/piControl/*MPI-ESM-MR*.nc'))
cubes = iris.load(files,'ocean_meridional_overturning_mass_streamfunction',callback = remove_anonymous)
max_strmfun = []
year = []
month = []
for cube in cubes:
	for i,sub_cube in enumerate(cube.slices(['latitude', 'depth'])):
		max_strmfun.append(np.max(sub_cube.data))
		coord = sub_cube.coord('time')
		year.append(np.array([coord.units.num2date(value).year for value in coord.points]))
		month.append(np.array([coord.units.num2date(value).month for value in coord.points]))

year = np.array(year)
month = np.array(month)
max_strmfun = np.array(max_strmfun)
date = year+month/12.0
date = date[:,0]
sorting = np.argsort(date)


N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=12.0 #years valkue should be '1.0/12.0'
low_cutoff=(12.0*40.0) #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
Wn_low=(low_cutoff/2.0)/timestep_between_values

#b, a = scipy.signal.butter(N, Wn_low, 'low')
#low_pass_timeseries = scipy.signal.filtfilt(b, a, max_strmfun[sorting])

b, a = scipy.signal.butter(3, 0.005, 'low')
#how many months is this filter???
low_pass_timeseries = scipy.signal.filtfilt(b, a, max_strmfun[sorting])

low_pass_timeseriesb = np.subtract(low_pass_timeseries,np.mean(low_pass_timeseries))

high_years = year[np.where(low_pass_timeseriesb >= 0.0)]

high_years = np.unique(high_years)

low_years = year[np.where(low_pass_timeseriesb <= 0.0)]

low_years = np.unique(low_years)


# plt.plot(date[sorting],max_strmfun[sorting])
# plt.plot(date[sorting],low_pass_timeseries,linewidth = 3)
# plt.show()

files = np.array(glob.glob(directory+'tos/piControl/*MPI-ESM-MR*.nc'))
cubes = iris.load(files,'sea_surface_temperature',callback = my_callback)

high_amo_data = cubes[0][0].data*0.0
low_amo_data = cubes[0][0].data*0.0
high_amo_count = 0.0
low_amo_count = 0.0

for j,cube in enumerate(cubes):
    print np.str(j)+' of '+np.str(len(cubes))
    cube = m2yr.monthly_to_yearly(cube)
    length = cube.shape[0]
    for i,sub_cube in enumerate(cube.slices(['cell index along second dimension', 'cell index along first dimension'])):
        print  np.str(i)+' of '+np.str(length)
        tmp_cube = sub_cube.copy()
        #copy for memory reasons
        coord = tmp_cube.coord('time')
        yr_tmp = (np.array([coord.units.num2date(value).year for value in coord.points]))[0]
        if np.in1d(np.array(yr_tmp), np.array(high_years))[0]:
            high_amo_data = np.sum([high_amo_data,tmp_cube.data],axis = 0, out=high_amo_data)
            high_amo_count += 1.0
        if np.in1d(np.array(yr_tmp), np.array(low_years))[0]:
            #low_amo_data += tmp_cube.data
            low_amo_data = np.sum([low_amo_data,tmp_cube.data],axis = 0, out=low_amo_data)
            low_amo_count += 1.0
    
high_amo_mean = np.divide(high_amo_data,high_amo_count)
cube_high_amo_mean = cubes[0][0].copy()
cube_high_amo_mean.data = high_amo_mean
    
low_amo_mean = np.divide(low_amo_data,low_amo_count)
cube_low_amo_mean = cubes[0][0].copy()
cube_low_amo_mean.data = low_amo_mean

#qplt.contourf(cube_high_amo_mean,50)
#plt.show()

'''
and precip
'''


files_pr = np.array(glob.glob(directory+'pr/piControl/*MPI-ESM-MR*.nc'))
cubes_pr = iris.load(files_pr)

high_amo_data_pr = cubes_pr[0][0].data*0.0
low_amo_data_pr = cubes_pr[0][0].data*0.0
high_amo_count_pr = 0.0
low_amo_count_pr = 0.0

for j,cube in enumerate(cubes_pr):
    print np.str(j)+' of '+np.str(len(cubes_pr))
    cube = m2yr.monthly_to_yearly(cube)
    length = cube.shape[0]
    for i,sub_cube in enumerate(cube.slices(['latitude', 'longitude'])):
        print  np.str(i)+' of '+np.str(length)
        tmp_cube = sub_cube.copy()
        coord = tmp_cube.coord('time')
        yr_tmp = (np.array([coord.units.num2date(value).year for value in coord.points]))[0]
        if np.in1d(np.array(yr_tmp), np.array(high_years))[0]:
            high_amo_data_pr = np.sum([high_amo_data_pr, tmp_cube.data],axis = 0, out=high_amo_data_pr)
            #high_amo_data_pr += sub_cube.data_pr
            high_amo_count_pr += 1.0
        else:
            low_amo_data_pr = np.sum([low_amo_data_pr, tmp_cube.data],axis = 0, out=low_amo_data_pr)
            #low_amo_data_pr += sub_cube.data_pr
            low_amo_count_pr += 1.0
    
high_amo_mean_pr = np.divide(high_amo_data_pr,high_amo_count_pr)
cube_high_amo_mean_pr = cubes_pr[0][0].copy()
cube_high_amo_mean_pr.data = high_amo_mean_pr
    
low_amo_mean_pr = np.divide(low_amo_data_pr,low_amo_count_pr)
cube_low_amo_mean_pr = cubes_pr[0][0].copy()
cube_low_amo_mean_pr.data = low_amo_mean_pr



'''
And plot the data
'''


sst = np.subtract(cube_high_amo_mean.data,cube_low_amo_mean.data)
plt.contourf(np.flipud(sst), (np.arange(100)/200.0)-0.25)
plt.show()

pr = iris.analysis.maths.np.subtract(cube_high_amo_mean_pr,cube_low_amo_mean_pr)
qplt.contourf(pr,60)
plt.gca().coastlines()
plt.show()
