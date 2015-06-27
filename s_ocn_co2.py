import iris
import glob
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
import iris.plot as iplt

'''

files = glob.glob('/home/ph290/data0/cmip5_data/regridded/fgco2*.nc')

cubes = []

for file in files:
    print file
    cube = iris.load_cube(file)
    cube = cube.collapsed('time',iris.analysis.MEAN)
    cubes.append(cube)

'''



fig = plt.figure(figsize=(6, 12))

for i,cube in enumerate(cubes):
    print files[i]
    if i == 6:
        cube = cube/12.0
    lon_west = -180
    lon_east = 180
    lat_south = -90
    lat_north = -20
    cube_region_tmp = cube.intersection(longitude=(lon_west, lon_east))
    cube_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))
    levels = np.linspace(-1e-9,1e-9)
    plt.subplot(8,1,i+1)
    plt.title(files[i].split('/')[-1].split('_')[1]+' air-sea CO2 flux',  fontsize=10)
    contour_result = iplt.contourf(cube_region,levels)
    plt.gca().coastlines()

#cbar = plt.colorbar(contour_result, orientation='horizontal')
#plt.show()
plt.savefig('/home/ph290/Documents/figures/co2.png')


