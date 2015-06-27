import numpy as np
import iris
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import cartopy.crs as ccrs
import cube_extract_region
import monthly_to_yearly
import contour_plot_irreular_grid

# def my_callback(cube, field, filename):
#     bad_coord1 = cube.coord(dimensions=1)
#     bad_coord2 = cube.coord(dimensions=2)
#     cube.remove_coord(bad_coord1)
#     cube.remove_coord(bad_coord2)
#     good_coord1 = cube.coords('latitude')
#     good_coord2 = cube.coords('longitude')
#     cube.add_dim_coord(good_coord1,data_dim=1)
#     cube.add_dim_coord(good_coord2,data_dim=2)

# cube = iris.load_cube(tos_files,callback=my_callback)



tos_files='/home/ph290/data1/cmip5_data/mpi_tos/tos*.nc'
#tos_files='/home/ph290/data1/cmip5_data//mpi_tos/tos_Omon_MPI-ESM-MR_piControl_r1i1p1_185001-189912.nc'
msftmyz_file='/home/ph290/data1/cmip5_data/mpi_msftmyz/msftmyz*.nc'
#msftmyz_file='/home/ph290/data1/cmip5_data/mpi_msftmyz/msftmyz_Omon_MPI-ESM-MR_piControl_r1i1p1_185001-189912.nc'
areacello='/home/ph290/data1/cmip5_data/areacello_fx_MPI-ESM-MR_1pctCO2_r0i0p0.nc'

areacello_cube=iris.load_cube(areacello)

tos_cube_tmp=iris.load_cube(tos_files)
tos_cube=tos_cube_tmp
#tos_cube = monthly_to_yearly.monthly_to_yearly(tos_cube_tmp)
msftmyz_cube=iris.load_cube(msftmyz_file)
msftmyz_cube_atlantic_tmp=msftmyz_cube[:,0,:,:]
msftmyz_cube_atlantic=msftmyz_cube_atlantic_tmp
#msftmyz_cube_atlantic = monthly_to_yearly.monthly_to_yearly(msftmyz_cube_atlantic_tmp)

moc=[]

for yz_slice in msftmyz_cube_atlantic.slices(['depth', 'latitude']):
   moc=np.append(moc,np.max(yz_slice.data))
   #so this now holds the max atlantic stream function for each month

#regrid data
#var_regridded, extent = iris.analysis.cartography.project(tos_cube[0], ccrs.PlateCarree())

min_lat=55
max_lat=58
min_lon=-10+360
max_lon=-6+360

# for testing...
# tos_cube_tmp=iris.load_cube(tos_files)
# tos_cube = monthly_to_yearly.monthly_to_yearly(tos_cube_tmp)
# cube=tos_cube
# cube2 = cube_extract_region.cube_extract_region(cube,min_lat,min_lon,max_lat,max_lon)
# var_regridded, extent = iris.analysis.cartography.project(cube2[0], ccrs.PlateCarree())
# qplt.contourf(var_regridded)
# plt.show()

#tmp_cube=tos_cube.copy()
tmp_cube=tos_cube.copy()
cube2 = cube_extract_region.cube_extract_region(tmp_cube,min_lat,min_lon,max_lat,max_lon)
cube_shape=cube2.shape

weights=areacello_cube.data
weights2=np.tile(weights,(cube_shape[0],1,1))
scotland_area_avged = cube2.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=weights2)

min_lat=66
max_lat=70
min_lon=-20+360
max_lon=-16+360

tmp_cube=tos_cube.copy()
cube3 = cube_extract_region.cube_extract_region(tmp_cube,min_lat,min_lon,max_lat,max_lon)
cube_shape=cube3.shape

weights=areacello_cube.data
weights2=np.tile(weights,(cube_shape[0],1,1))
iceland_area_avged = cube3.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=weights2)

plt.scatter(scotland_area_avged.data-iceland_area_avged.data,moc)
plt.show()

#qplt.contourf(cube2[2])
#plt.show()

#contour_plot_irreular_grid.contour_plot_irreular_grid(cube3[0])
#contour_plot_irreular_grid.contour_plot_irreular_grid(tos_cube[0])
