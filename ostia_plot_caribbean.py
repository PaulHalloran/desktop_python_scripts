import numpy as np
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis
import iris.coords as icoords
import cartopy.crs as ccrs
import cartopy

# def my_callback(cube, field, filename):
#     if cube.ndim == 3 and cube.shape[0] == 1:
#         # I would normally slice the data out here (i.e. do cube = cube[0, ...] but that doesn't work
#         # in a callback at the moment - I'll be making a ticket for that!). Instead, we manually remove
#         # the first dimension which has a length of 1.
#         bad_coord = cube.coord(dimensions=0)
#         cube.remove_coord(bad_coord)
#         cube.add_aux_coord(bad_coord)
#         dc = cube._dim_coords_and_dims
#         cube._dim_coords_and_dims = [(coord, dim-1) for coord, dim in dc if dim > 0]
#         cube._data = cube.data[0, ...]
#     # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
#     cube.attributes.pop('start_date')
#     cube.attributes.pop('stop_date')
#     cube.attributes.pop('stop_time')
#     # XXX I didn't figure out which were bad, so I removed them all. 
#     cube.attributes = {}


def my_callback(cube, field, filename):
    #this is faster than the commented out callback above (because it does not load in all of the data), but keeping the above because more descriptive
    # Remove the first dimension (time) if it has a length of one.
    # This allows iris' merge to join similar cubes together.
    if cube.ndim == 3 and cube.shape[0] == 1:
        cube = cube[0, ...]

    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('start_date')
    cube.attributes.pop('stop_date')
    cube.attributes.pop('stop_time')
    cube.attributes.pop('history')
    return cube

def main():
    dir='/Users/ph290/Public/mo_data/ostia/' # on ELD140
    filename = dir + '*.nc'

    cube = iris.load_cube(filename,'sea_surface_temperature',callback=my_callback)
    #reads in data using a special callback, because it is a nasty netcdf file
    cube.data=cube.data-273.15


    sst_mean = cube.collapsed('time', iris.analysis.MEAN)
    #average all 12 months together
    sst_stdev=cube.collapsed('time', iris.analysis.STD_DEV)

    caribbean = iris.Constraint(
                                    longitude=lambda v: 260 <= v <= 320,
                                    latitude=lambda v: 0 <= v <= 40,
                                    name='sea_surface_temperature'
                                    )

    caribbean_sst_mean = sst_mean.extract(caribbean)
    caribbean_sst_stdev = sst_stdev.extract(caribbean)
    #extract the Caribbean region

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    data=caribbean_sst_mean.data
    data2=caribbean_sst_stdev.data
    lons = caribbean_sst_mean.coord('longitude').points
    lats = caribbean_sst_mean.coord('latitude').points
    lo = np.floor(data.min())
    hi = np.ceil(data.max())
    levels = np.linspace(lo,hi,100)
    lo2 = np.floor(data2.min())
    hi2 = np.ceil(data2.max())
    levels2 = np.linspace(lo2,5,10)
    cube_label = 'latitude: %s' % caribbean_sst_mean.coord('latitude').points
    contour=plt.contourf(lons, lats, data,transform=ccrs.PlateCarree(),levels=levels,xlabel=cube_label)
    #filled contour the annually averaged temperature
    contour2=plt.contour(lons, lats, data2,transform=ccrs.PlateCarree(),levels=levels2,colors='k')
    #contour the standard deviations
    plt.clabel(contour2, inline=0.5, fontsize=12,fmt='%1.1f' )
    ax.add_feature(cartopy.feature.LAND)
    ax.coastlines()
    ax.add_feature(cartopy.feature.RIVERS)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    #ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    cbar = plt.colorbar(contour, ticks=np.linspace(lo,hi,7), orientation='horizontal')
    cbar.set_label('Sea Surface Temperature ($^\circ$C)')
    # enable axis ticks
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    # fix 10-degree increments and don't clip range
    plt.locator_params(steps=(1,10), tight=False)
    # add gridlines
    plt.grid(True)
    # add axis labels
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    #plt.show()
    plt.savefig('/home/h04/hador/public_html/twiki_figures/carib_sst_and_stdev.png')


if __name__ == '__main__':
    main()

