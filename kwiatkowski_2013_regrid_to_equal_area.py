import iris
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.analysis.cartography
import glob

files = glob.glob('/home/ph290/data1/lester_data/*.nc')

for file_name in files:
    tmp = iris.load_cube(file_name)
    #tmp=sst[0]

    fig=plt.figure()
    qplt.contourf(tmp[0])
    plt.show()

    tmp_regridded, extent = iris.analysis.cartography.project(tmp, ccrs.Mollweide())

    plt.contourf(tmp_regridded[0].data)
    plt.show()

    out = file_name.split('/')
    iris.fileformats.netcdf.save(tmp_regridded, '/home/ph290/Desktop/equal_'+out[-1], netcdf_format='NETCDF3_CLASSIC')
