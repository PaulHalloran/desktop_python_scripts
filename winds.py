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
cube1 = iris.load_cube('/home/ph290/Documents/teaching/zonal_wind.nc')[12]
cube2 = iris.load_cube('/home/ph290/Documents/teaching/meridional_wind.nc')[12]
cube3 = iris.analysis.maths.multiply(cube1,cube2)


lats = cube1.coord('latitude').points
lons = cube1.coord('longitude').points
lons, lats = np.meshgrid(lons, lats)

plt.figure()
# ax = plt.axes(projection=ccrs.Mollweide())
# ax.coastlines()
plt.quiver(lats,lons,cube1.data,(cube2.data),color='k', units='width',
            width = 0.004,scale = 200,headwidth = 2.0, edgecolors=('k'))
# ax.add_feature(cartopy.feature.LAND)
#note - this step is slow (perhaps a minute - don't give up!)
# cb1 = plt.colorbar(cb, orientation="vertical")
# cb1.set_label('Sea surface temperature ($^o$C)')
plt.show()
# plt.savefig('/home/ph290/Documents/figures/currents_sst.pdf')
