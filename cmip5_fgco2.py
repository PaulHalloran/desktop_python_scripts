'''
This script reads in surface ocean CO2 flux from cmip5 models and produces a global mean without having to regrid (i.e. on thier natural grids)
'''

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
import iris.coord_categorisation


def monthly_to_yearly(cube):
    try:
        iris.coord_categorisation.add_year(cube, 'time', name='year2')
    except ValueError:
        cube_tmp = cube.aggregated_by('year2', iris.analysis.MEAN)
        return cube_tmp

def my_callback(cube, field,files):
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('history')
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    #if np.size(cube) > 1:
    #cube = iris.experimental.concatenate.concatenate(cube)
    return cube


directory = '/data/data0/ph290/cmip5_data/fgco2/rcp85/'
files = np.array(glob.glob(directory+'*.nc'))

models = []
for file in files:
    models.append(file.split('/')[-1].split('_')[2])

models_unique = np.unique(np.array(models))

'''
read in model data
'''

loaded_models = []
cubes = []
for i,model in enumerate(models_unique):
    print i
    file = glob.glob(directory+'*'+model+'*.nc')
    cube = iris.load(file,'surface_downward_mass_flux_of_carbon_dioxide_expressed_as_carbon',callback=my_callback)
    if len(cube) > 1:
        cube = iris.experimental.concatenate.concatenate(cube)
    if len(cube) == 1:
        cubes.append(cube[0])
        loaded_models.append(model)
        print model

'''
read in aeracello files
'''

aco_directory = '/home/ph290/data1/cmip5_data/areacello_files/'

aco_cubes = []
for i,model in enumerate(loaded_models):
    print i
    file = glob.glob(aco_directory+'*'+model+'*.nc')
    cube = iris.load(file[0])
    aco_cubes.append(cube[0])
 


global_mean = []
for i,cube in enumerate(cubes):
    #if loaded_models[i] not 'MPI-ESM-LR':
    print i
    shape = cube.shape
    aco = np.tile(aco_cubes[i].data,(shape[0],1,1))
    shape2 = aco.shape
    if not shape == shape2:
        if not cube.coord('latitude').has_bounds():
            cube.coord('latitude').guess_bounds()
        if not cube.coord('longitude').has_bounds():
            cube.coord('longitude').guess_bounds()
        aco = iris.analysis.cartography.area_weights(cube)   
    try:
        global_mean.append(cube.collapsed([cube.coords()[1].standard_name,cube.coords()[2].standard_name],iris.analysis.MEAN,weights = aco))
    except TypeError:
        global_mean.append(cube.collapsed([cube.coords()[1].long_name,cube.coords()[2].long_name],iris.analysis.MEAN,weights = aco))

#linestyles = ['-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':']
linestyles = ['-', '--', '-', '--','-', '--', '-', '--','-', '--', '-', '--','-', '--', '-', '--','-', '--', '-', '--','-', '--', '-', '--']

for i,model in enumerate(loaded_models):
    line = qplt.plot(monthly_to_yearly(global_mean[i]))
    plt.setp(line, linestyle=linestyles[i],linewidth = 2)

plt.show()

for j,model in enumerate(loaded_models):
    line = plt.plot([0,1],[j+1,j+1])
    plt.text(1.2, j+0.8, model,fontsize=12)
    plt.setp(line, linestyle=linestyles[j],linewidth = 2)
    plt.xlim([0,2])
    plt.ylim([0,18])

plt.show()
