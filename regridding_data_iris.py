import iris
import iris.analysis.cartography
import iris.analysis
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.coords
import iris.quickplot as qplt
import cartopy.crs as ccrs

file_name='/project/champ/data/cmip5/output1/MPI-M/MPI-ESM-MR/historical/yr/ocnBgchem/Oyr/r1i1p1/v20120503/talk/talk_Oyr_MPI-ESM-MR_historical_r1i1p1_1860-1869.nc'

cube=iris.load_cube(file_name)
surface_slice = cube.extract(iris.Constraint(depth=0))
#extract just the surface level

#start by making a regular 180 by 360 cube to regrid to:
latitude = iris.coords.DimCoord(range(-90, 90, 1), standard_name='latitude', units='degrees')
longitude =  iris.coords.DimCoord(range(0, 360, 1), standard_name='longitude', units='degrees')
regridding_cube = iris.cube.Cube(np.zeros((180, 360), np.float32),dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

regridded_cube=iris.analysis.interpolate.regrid(surface_slice, regridding_cube, mode='bilinear')


#plotting

qplt.contourf(cube[0,0,:,:], 20)
#plt.gca().coastlines()
plt.show()
