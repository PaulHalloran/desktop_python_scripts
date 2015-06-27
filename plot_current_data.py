import iris
import matplotlib.pyplot as plt
import numpy as np
import iris.plot as iplt
import cartopy.crs as ccrs
import cartopy
import matplotlib.cm as mpl_cm

'''
Read in current data
'''
cube1 = iris.load_cube('/home/ph290/Documents/teaching/zonal_current.nc')
cube2 = iris.load_cube('/home/ph290/Documents/teaching/meridional_current.nc')
cube3 = iris.analysis.maths.exponentiate(iris.analysis.maths.add(iris.analysis.maths.multiply(cube1,cube1),iris.analysis.maths.multiply(cube2,cube2)),0.5)

'''
Read in SST and p-e data
'''
sst = iris.load_cube('/home/ph290/Documents/teaching/temperature_annual_1deg.nc','Statistical Mean')
p_e = iris.load_cube('/home/ph290/Documents/teaching/evap_precip_annual_HOAPS32.nc')

'''
Specify resolution of grid downscaling current data to (to avoid too many current arrows)
'''
spacing_deg = 8
lat_array = range(-90, 90, spacing_deg)
lon_array = range(0, 360, spacing_deg)

'''
Produce a regridding cube
'''
latitude = iris.coords.DimCoord(lat_array, standard_name='latitude',
                    units='degrees')
longitude = iris.coords.DimCoord(lon_array, standard_name='longitude',
                     units='degrees')
cube = iris.cube.Cube(np.zeros((np.ceil(180.0/spacing_deg), 360.0/spacing_deg), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])


'''
the following is to set the data below the masked areas (i.e. land) to nan, so tha the regridding does not use that information and mess things up
'''
test1 = np.ma.where(cube1.data.data <= -5.0)
test2 = np.ma.where(cube2.data.data <= -5.0)
cube1.data.data[test1] = np.zeros(np.shape(test1)[1])+np.nan
cube2.data.data[test2] = np.zeros(np.shape(test2)[1])+np.nan

'''
Downscales the number of current grid-points to thin our current arrows
'''
cube1b =  iris.analysis.interpolate.regrid(cube1,cube)
cube2b =  iris.analysis.interpolate.regrid(cube2,cube)

lats = sst[0][0].coord('latitude').points
lons = sst[0][0].coord('longitude').points
lons, lats = np.meshgrid(lons, lats)

# plt.figure()
# ax = plt.axes(projection=ccrs.Mollweide())
# cb = ax.contourf(lons,lats,sst[0][0].data,50,transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.quiver(lon_array,lat_array,cube1b.data,cube2b.data,color='k', units='width',
#             width = 0.004,scale = 5,headwidth = 2.0, edgecolors=('k'),transform=ccrs.PlateCarree() )
# ax.add_feature(cartopy.feature.LAND)
# #note - this step is slow (perhaps a minute - don't give up!)
# cb1 = plt.colorbar(cb, orientation="vertical")
# cb1.set_label('Sea surface temperature ($^o$C)')
# #plt.show()
# plt.savefig('/home/ph290/Documents/figures/currents_sst.pdf')

lats = p_e.coord('latitude').points
lons = p_e.coord('longitude').points
lons, lats = np.meshgrid(lons, lats)

brewer_cmap = mpl_cm.get_cmap('brewer_RdBu_11')

# plt.figure()
# ax = plt.axes(projection=ccrs.Mollweide())
# cb = ax.contourf(lons,lats,p_e.data,np.linspace(-10,10,51.0),cmap=brewer_cmap,transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.quiver(lon_array,lat_array,cube1b.data,cube2b.data,color='k', units='width',
#             width = 0.004,scale = 5,headwidth = 2.0, edgecolors=('k'),transform=ccrs.PlateCarree() )
# ax.add_feature(cartopy.feature.LAND)
# #note - this step is slow (perhaps a minute - don't give up!)
# cb1 = plt.colorbar(cb, orientation="vertical")
# cb1.set_label('Freshwater flux (surface upwards) mm/day')
# #plt.show()
# plt.savefig('/home/ph290/Documents/figures/currents_p_e.pdf')


lats = cube3.coord('latitude').points
lons = cube3.coord('longitude').points
lons, lats = np.meshgrid(lons, lats)

plt.figure()
ax = plt.axes(projection=ccrs.Mollweide())
cb = ax.contourf(lons,lats,cube3.data,np.linspace(0.0,0.75,51),transform=ccrs.PlateCarree())
ax.coastlines()
# ax.quiver(lon_array,lat_array,cube1b.data,cube2b.data,color='k', units='width',
#             width = 0.004,scale = 5,headwidth = 2.0, edgecolors=('k'),transform=ccrs.PlateCarree() )
# ax.add_feature(cartopy.feature.LAND)
#note - this step is slow (perhaps a minute - don't give up!)
cb1 = plt.colorbar(cb, orientation="vertical")
cb1.set_label('current speed (m/s)')
# plt.show()
plt.savefig('/home/ph290/Documents/figures/current_intensity.pdf')

