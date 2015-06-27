import numpy as np
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis
import iris.coords as icoords


def main():
    dir='/data/local2/hador/ostia_reanalysis/' # ELD140
    filename = dir + '*.nc'

    cube = iris.load_cube(filename,'sea_surface_temperature',callback=my_callback)
    #reads in data using a special callback, because it is a nasty netcdf file

    sst_mean = cube.collapsed('time', iris.analysis.MEAN)
    #average all 12 months together

    caribbean = iris.Constraint(
                                    longitude=lambda v: 260 <= v <= 320,
                                    latitude=lambda v: 0 <= v <= 40,
                                    name='sea_surface_temperature'
                                    )

    caribbean_sst_mean = sst_mean.extract(caribbean)
    #extract the Caribbean region
    
    plt.figure()
    contour=qplt.contourf(caribbean_sst_mean, 50)
    contour=qplt.contour(caribbean_sst_mean, 5,colors='k')
    plt.clabel(contour, inline=1, fontsize=10,fmt='%1.1f' )
    plt.gca().coastlines()
    #plt.gca().set_extent((-100,-60,0,40))
    plt.show()



def my_callback(cube, field, filename):
    if cube.ndim == 3 and cube.shape[0] == 1:
        # I would normally slice the data out here (i.e. do cube = cube[0, ...] but that doesn't work
        # in a callback at the moment - I'll be making a ticket for that!). Instead, we manually remove
        # the first dimension which has a length of 1.
        bad_coord = cube.coord(dimensions=0)
        cube.remove_coord(bad_coord)
        cube.add_aux_coord(bad_coord)
        dc = cube._dim_coords_and_dims
        cube._dim_coords_and_dims = [(coord, dim-1) for coord, dim in dc if dim > 0]
        cube._data = cube.data[0, ...]
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('start_date')
    cube.attributes.pop('stop_date')
    cube.attributes.pop('stop_time')
    # XXX I didn't figure out which were bad, so I removed them all. 
    cube.attributes = {}

if __name__ == '__main__':
    main()
