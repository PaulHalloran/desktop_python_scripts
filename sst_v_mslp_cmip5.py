import numpy as np
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
import scipy
from scipy import signal

def analysis(t_cube,p_cube):
    points1 = t_cube.coord('time').points
    points2 = p_cube.coord('time').points
    test = np.sum(points2 - points2)
    if test == 0:
        try:
            iris.coord_categorisation.add_year(t_cube, 'time', name='year2')
            iris.coord_categorisation.add_season_number(t_cube, 'time', name='season')
            iris.coord_categorisation.add_year(p_cube, 'time', name='year2')
            iris.coord_categorisation.add_season_number(p_cube, 'time', name='season')
        except ValueError:
            pass
        time_constraint = iris.Constraint(season = 2)
        cube_tmp = t_cube.extract(time_constraint)
        #this just extracts summer months (jja)
        t_cube_jja = cube_tmp.aggregated_by('year2', iris.analysis.MEAN)
        cube_tmp = p_cube.extract(time_constraint)
        #this just extracts summer months (jja)
        p_cube_jja = cube_tmp.aggregated_by('year2', iris.analysis.MEAN)
        site_lat_lon = np.array([66.0,-20.0])
        lat = t_cube_jja.coord('latitude')
        lon = t_cube_jja.coord('longitude')
        lat_coord1 = lat.nearest_neighbour_index(site_lat_lon[0])
        lon_coord1 = lon.nearest_neighbour_index(site_lat_lon[1])
        site_timeseries_t = t_cube_jja.data[:,lat_coord1,lon_coord1]
        site_timeseries_t_detrended = scipy.signal.detrend(site_timeseries_t)
        ts_stdev = site_timeseries_t_detrended.std()
        high = np.where(site_timeseries_t_detrended >= ts_stdev)
        low = np.where(site_timeseries_t_detrended <= ts_stdev*(-1.0))
        p_cube_jja.data = scipy.signal.detrend(p_cube_jja.data,axis=0)
        high_mslp = p_cube_jja[high].collapsed('time',iris.analysis.MEAN)
        low_mslp = p_cube_jja[low].collapsed('time',iris.analysis.MEAN)
        return high_mslp,low_mslp
    else:
        print 'year order/number not the same between temperature and pressure fields'

def get_details(directory,files):
    model_name=[]
    for file in files:
        model_name.append(file.split('_')[3])
    model_name = np.array(model_name)
    model_name_unique = np.unique(model_name)
    ens_member=[]
    for file in files:
        tmp = file.split('_')[5]
        ens_member.append(np.int(tmp[1]))
    ens_member = np.array(ens_member)
    file_year=[]
    for file in files:
        tmp = file.split('_')[6]
        file_year.append(np.int(tmp.split('.')[0].split('-')[0]))
    file_year = np.array(file_year)
    return model_name_unique,model_name,ens_member,file_year

def read_in_data(model_name_unique,model_name,ens_member,files):
    all_cubes=[]
    for model in model_name_unique:
        print model
        loc = np.where((model_name == model))
        loc2 = np.where(ens_member[loc[0]] == np.min(ens_member[loc[0]]))
        files_tmp=files[loc[0][loc2[0]]]
        cubes = iris.load_raw(files_tmp,callback=my_callback)
        cube = iris.experimental.concatenate.concatenate(cubes)
        all_cubes.append(cube[0])
    return all_cubes

def my_callback(cube, field,files_tmp):
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('history')
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    # if np.size(cube) > 1:
    #     cube = iris.experimental.concatenate.concatenate(cube)
    return cube

def monthly_to_yearly(cube):
    if np.size(cube._aux_coords_and_dims) < 2:
        iris.coord_categorisation.add_year(cube, 'time', name='year2')
    cube_tmp = cube.aggregated_by('year2', iris.analysis.MEAN)
    return cube_tmp


directory1 = '/data/data0/ph290/cmip5_data/picontrol/tas/'

files1=glob.glob(directory1+'tas*.nc')
files1=np.array(files1)

directory2 = '/data/data0/ph290/cmip5_data/picontrol/mslp/'

files2=glob.glob(directory2+'psl*.nc')
files2=np.array(files2)

'''
get info about files amd read in data
'''

tas_model_name_unique,tas_model_name,tas_ens_member,tas_file_year = get_details(directory1,files1)
tas_cubes = read_in_data(tas_model_name_unique,tas_model_name,tas_ens_member,files1)

mslp_model_name_unique,mslp_model_name,mslp_ens_member,mslp_file_year = get_details(directory2,files2)
mslp_cubes = read_in_data(mslp_model_name_unique,mslp_model_name,mslp_ens_member,files2)

'''
analysis:
'''

model = 'MPI-ESM-LR'
t_cube = tas_cubes[np.where(tas_model_name_unique == model)[0][0]]
p_cube = mslp_cubes[np.where(mslp_model_name_unique == model)[0][0]]

high_mslp,low_mslp = analysis(t_cube,p_cube)

plt.figure()
qplt.contourf(high_mslp,30)
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/'+model+'_warm.png')

plt.figure()
qplt.contourf(low_mslp,30)
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/'+model+'_cool.png')

model = 'IPSL-CM5A-LR'
t_cube = tas_cubes[np.where(tas_model_name_unique == model)[0][0]]
p_cube = mslp_cubes[np.where(mslp_model_name_unique == model)[0][0]]

high_mslp,low_mslp = analysis(t_cube,p_cube)

plt.figure()
qplt.contourf(high_mslp,30)
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/'+model+'_warm.png')

plt.figure()
qplt.contourf(low_mslp,30)
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/'+model+'_cool.png')


model = ''

t_cube = tas_cubes[np.where(tas_model_name_unique == model)[0][0]]
p_cube = mslp_cubes[np.where(mslp_model_name_unique == model)[0][0]]

high_mslp,low_mslp = analysis(t_cube,p_cube)

plt.figure()
qplt.contourf(high_mslp,30)
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/'+model+'_warm.png')

plt.figure()
qplt.contourf(low_mslp,30)
plt.gca().coastlines()
plt.savefig('/home/ph290/Documents/figures/'+model+'_cool.png')
