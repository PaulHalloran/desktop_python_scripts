import iris
import matplotlib.pyplot as plt
import glob
import numpy as np
import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import uuid
import iris.quickplot as qplt
import cartopy
from iris.analysis.geometry import geometry_area_weights
from cartopy.io.shapereader import Reader, natural_earth
from shapely.ops import cascaded_union

def unique_models(all_files):
    model = []
    for file in all_files:
        model.append(file.split('/')[8])
    return np.unique(np.array(model))


'''
EDIT THE FOLLOWING TEXT
'''

input_directory = '/media/usb_external1/ocmip2/c14/'
resolution = 5 #degrees (for regridding - not low resolution of OCMIP2 models)

'''
Main bit of code follows...
'''

all_files = glob.glob('/home/ph290/data0/ocmip/dods.ipsl.jussieu.fr/ocmip/phase2/*/Abiotic/hist/*.nc')

data = np.genfromtxt('/home/ph290/data0/misc_data/c14nth.dat',skip_header = 4)
atm14_yr = data[:,0]
atm14_data = data[:,1]

models = unique_models(all_files)

models2 = models[np.array([0,2,3,4,5,6,7,9,10,11])]
#these are the models that have regridded properly!

cubes = []

'''

for model in models2:
#	cubes.append(iris.load_cube(input_directory+model+'_d14c_hist.nc'))
	cube1 = iris.load_cube(input_directory+model+'_d14c_hist.nc')
	cube2 = iris.load_cube(input_directory+model+'_dpco2_hist.nc')

	#cube1.coord('latitude').guess_bounds()
	#cube1.coord('longitude').guess_bounds()
	#cube2.coord('latitude').guess_bounds()
	#cube2.coord('longitude').guess_bounds()

	lon_west = -75.0
	lon_east = -7.5
	lat_south = 45.0
	lat_north = 60.0 

	cube_region_tmp = cube1.intersection(longitude=(lon_west, lon_east))
	cube1_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))

	cube_region_tmp = cube2.intersection(longitude=(lon_west, lon_east))
	cube2_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))

	grid_areas = iris.analysis.cartography.area_weights(cube1_region)
	area_avged_cube1 = cube1_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
	grid_areas = iris.analysis.cartography.area_weights(cube1_region)
	area_avged_cube2 = cube2_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(area_avged_cube1.coord('time').points[0:-1], area_avged_cube1.data[1::]-area_avged_cube1.data[0:-1])
	ax2 = ax1.twinx()
	ax2.plot(area_avged_cube2.coord('time').points[0:-1], area_avged_cube2.data[1::]-area_avged_cube2.data[0:-1],'r')
	plt.show()

'''


model = models2[1]
cube1 = iris.load_cube(input_directory+model+'_d14c_hist.nc')
cube2 = iris.load_cube(input_directory+model+'_dpco2_hist.nc')

#cube1.coord('latitude').guess_bounds()
#cube1.coord('longitude').guess_bounds()
#cube2.coord('latitude').guess_bounds()
#cube2.coord('longitude').guess_bounds()

lon_west = -75.0
lon_east = -7.5
lat_south = 45.0
lat_north = 60.0 

cube_region_tmp = cube1.intersection(longitude=(lon_west, lon_east))
cube1_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))

cube_region_tmp = cube2.intersection(longitude=(lon_west, lon_east))
cube2_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))

grid_areas = iris.analysis.cartography.area_weights(cube1_region)
area_avged_cube1 = cube1_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
grid_areas = iris.analysis.cartography.area_weights(cube1_region)
area_avged_cube2 = cube2_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(area_avged_cube1.coord('time').points[0:-1], area_avged_cube1.data[1::]-area_avged_cube1.data[0:-1])
ax2 = ax1.twinx()
ax2.plot(area_avged_cube2.coord('time').points[0:-1], area_avged_cube2.data[1::]-area_avged_cube2.data[0:-1],'r')
ax3 = ax2.twinx()
plt.plot(atm14_yr,atm14_data,'y')
plt.xlim([1950,2000])
plt.show()
