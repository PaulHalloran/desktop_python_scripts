import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import matplotlib.cm as mpl_cm
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

def remove_anonymous(cube, field, filename):
# this only loads in the region relating to the atlantic
        cube.attributes.pop('creation_date')
        cube.attributes.pop('tracking_id')
        cube = cube[:,0,:,:]
        return cube


def regridding_unstructured(cube):
    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points
    tmp_shape_lats = lats.shape
    tmp_shape_lons = lons.shape
    if len(tmp_shape_lats) == 1:
        cube2 = iris.cube.Cube(np.zeros((180, 360), np.float32),standard_name='air_temperature', long_name='air_temperature', var_name='tas', units='kg m-2 s-1',dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
        out =  iris.analysis.interpolate.regrid(cube, cube2)
        return out.data,out.data,out.data
    else:
        data = np.array(cube.data)
        lats2 = np.reshape(lats,lats.shape[0]*lats.shape[1])
        lons2 = np.reshape(lons,lons.shape[0]*lons.shape[1])
        data2 = np.reshape(data,data.shape[0]*data.shape[1])
        yi = np.linspace(-90.0,90.0,180.0)
        xi = np.linspace(0.0,360.0,360.0)
        zi = ml.griddata(lons2,lats2,data2,xi,yi)
        return xi,yi,zi

def cube_extract_region(cube,min_lat,min_lon,max_lat,max_lon):
    region = iris.Constraint(longitude=lambda v: min_lon <= v <= max_lon,latitude=lambda v: min_lat <= v <= max_lat)
    return cube.extract(region)



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
    return iris.load(out_filename)



def model_names(directory):
	files = glob.glob(directory+'*uo*piControl*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models

def model_names2(directory):
	files = glob.glob(directory+'*.nc')
	models_tmp = []
	for file in files:
		print file
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[2])
			models = np.unique(models_tmp)
	return models

amoc_dir = '/home/ph290/data0/cmip5_data/msftmyz/piControl/'

models_1 = model_names('/media/usb_external1/cmip5/gulf_stream_analysis/regridded/')
models_2 = model_names2(amoc_dir)
models = np.intersect1d(models_1,models_2)
remove = np.where(models == 'inmcm4')
#remove inmcm4 because goes funny in the regridding...
models = list(models)
models.pop(remove[0])
models = np.array(models)


runs = 'piControl'
variables = 'uo'

#for each model calculate a timeseried of gulf stream latutide at 45-55W
models_gs_lat = [] #timeseries for each model of teh gulf stream latitude
models_years = [] #the years for each model correstonding to the above

cmip5_max_strmfun = []
cmip5_year = []

for model in models:
    print model+' calculating max NA stream function'
    files = np.array(glob.glob(amoc_dir+'*_'+model+'_'+'*.nc'))
    cubes = iris.load(files,'ocean_meridional_overturning_mass_streamfunction',callback = remove_anonymous)
    max_strmfun = []
    year = []
    month = []
    for cube_tmp in cubes:
        cube = cube_tmp.copy()
        cube = m2yr.monthly_to_yearly(cube)
        for i,sub_cube in enumerate(cube.slices(['latitude', 'depth'])):
		sub_cube_n_hem = sub_cube.extract(iris.Constraint(latitude = lambda cell: cell > 0))
		max_strmfun.append(np.max(sub_cube_n_hem.data))
		coord = sub_cube_n_hem.coord('time')
		year.append(np.array([coord.units.num2date(value).year for value in coord.points])[0])
    cmip5_max_strmfun.append(np.array(max_strmfun))
    cmip5_year.append(np.array(year))



cmip5_max_strmfun = np.array(cmip5_max_strmfun)
cmip5_year = np.array(cmip5_year)


for model in models:
	print model+' calculating gulf stream possition'
	piControl_uo_file = '/media/usb_external1/cmip5/gulf_stream_analysis/regridded/'+model+'_'+variables+'_'+runs+'_regridded.nc'
	piControl_uo_cube = iris.load_cube(piControl_uo_file)
	coord_names = [coord.name() for coord in piControl_uo_cube.coords()]
	test1 = np.size(np.where(np.array(coord_names) == 'ocean sigma coordinate'))
	test1b = np.size(np.where(np.array(coord_names) == 'ocean sigma over z coordinate'))
	if test1 == 1:
		piControl_uo_cube.coord('ocean sigma coordinate').long_name = 'depth'


	if test1b == 1:
		piControl_uo_cube.coord('ocean sigma over z coordinate').long_name = 'depth'


	if np.size(piControl_uo_cube.coords()) >= 4:
		piControl_uo_cube = piControl_uo_cube.extract(iris.Constraint(depth = 0))


	piControl_uo_cube_50w = piControl_uo_cube.extract(iris.Constraint(longitude = lambda cell: 360-45.0 < cell < 360-35.0))
	piControl_uo_cube_50w = piControl_uo_cube_50w.extract(iris.Constraint(latitude = lambda cell: 25 < cell < 60))
	piControl_uo_cube_50w = piControl_uo_cube_50w.collapsed('longitude',iris.analysis.MEAN)
	latitudes = piControl_uo_cube_50w.coord('latitude').points
	coord = piControl_uo_cube_50w.coord('time')
	years = np.array([coord.units.num2date(value).year for value in coord.points])
	data = piControl_uo_cube_50w.data
	max_current = np.max(data,axis = 1)
	gulf_stream_lat = np.empty(years.size)
	for i in range(years.size):
		gulf_stream_lat[i] = latitudes[np.where(data[i,:] == max_current[i])][0]
	mean_gs_lat = gulf_stream_lat.mean()
	models_years.append(np.array(years))
	models_gs_lat.append(np.array(gulf_stream_lat))

models_years = np.array(models_years)
models_gs_lat = np.array(models_gs_lat)

common_years = []
strmfun_common_yrs = []
gs_common_yrs = []
model_no = []

for j,model in enumerate(models):
	strmfun = cmip5_max_strmfun[j]
	gs = models_gs_lat[j]
	yr1 = models_years[j]
	yr2 = cmip5_year[j]
	for i,year in enumerate(yr1):
		loc = np.where(year == yr2)
		if np.size(loc) > 0:
			common_years.append(year)
			strmfun_common_yrs.append(strmfun[loc[0][0]])
			gs_common_yrs.append(gs[i])
			model_no.append(j)

common_years = np.array(common_years)
strmfun_common_yrs = np.array(strmfun_common_yrs)
gs_common_yrs = np.array(gs_common_yrs)
model_no = np.array(model_no)

colours=['k','r','g','b','y']

'''
I need to think about this, but I think it should be plotting amoc strength v gulf stream latitude for these different models...
'''

model_unit_correction = np.where(models == 'CNRM-CM5')
x = np.where(model_no == model_unit_correction[0])
strmfun_common_yrs2 = strmfun_common_yrs.copy()
strmfun_common_yrs2[x] = strmfun_common_yrs[x] *1.0e6

fig = plt.figure()
for i in range(common_years.size):
	x = np.mean(strmfun_common_yrs2[np.where(model_no == model_no[i])])
	plt_x = strmfun_common_yrs2[i]/x
	y = np.mean(gs_common_yrs[np.where(model_no == model_no[i])])
	plt_y = gs_common_yrs[i]/y
	plt.scatter(plt_x,plt_y,color = colours[model_no[i]])

plt.xlabel('normalised max. atlantic stream function')
plt.ylabel('normalised latitude of gulf stream')
plt.title('diff models diff colours (control run)')
plt.savefig('/home/ph290/Documents/figures/gulfstream_analysis/amoc_v_gulf_stream.png')
#plt.show()
plt.close("all")



