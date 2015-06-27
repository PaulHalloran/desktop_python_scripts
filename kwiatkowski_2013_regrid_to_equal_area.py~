import iris
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.analysis.cartography

file_name = '/home/ph290/data1/observations/hadisst/HadISST_sst.nc'
sst = iris.load_cube(file_name)
tmp=sst[0]


tmp_regridded, extent = iris.analysis.cartography.project(tmp, ccrs.Mollweide())

plt.contourf(tmp_regridded.data)
plt.show()


iris.fileformats.netcdf.save(tmp_regridded, '/home/ph290/Desktop/hadisst_equal_area.nc', netcdf_format='NETCDF3_CLASSIC')
