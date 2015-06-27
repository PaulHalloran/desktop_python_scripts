import iris
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt
import cartopy
import matplotlib.cm as mpl_cm

bathy_cube = iris.load_cube('/home/ph290/data0/misc_data/ETOPO1_Bed_c_gmt4.grd')

lon_west = -60.0
lon_east = 0
lat_south = 20.0
lat_north = 90.0 

cube_region_tmp = bathy_cube.intersection(longitude=(lon_west, lon_east))
bathy_cube_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))


co2_cube = iris.load_cube('/home/ph290/data1/observations/SOCAT_tracks_gridded_month_clim_v2.nc','fCO2 mean - per cruise weighted')
co2_cube_JJA = co2_cube[5:8]
co2_cube_mean = co2_cube_JJA.collapsed('time',iris.analysis.MEAN)


cube_region_tmp = co2_cube_mean.intersection(longitude=(lon_west, lon_east))
co2_cube_mean_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))

plt.close('all')

plt.figure(dpi = 600)
cmap1 = mpl_cm.get_cmap('seismic')
cmap2 = mpl_cm.get_cmap('terrain')
cmap2 = mpl_cm.get_cmap('gist_rainbow')

land_110m = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m',
                                        edgecolor='face',
                                        facecolor = '#d3d3d3')

ax = plt.axes(projection=cartopy.crs.PlateCarree())
c = iplt.contourf(co2_cube_mean_region,np.linspace(150,500,31),cmap=cmap1)
iplt.contour(bathy_cube_region,np.linspace(-150,0,4),colors = 'k')
ax.add_feature(land_110m)
plt.colorbar(c)
plt.savefig('/home/ph290/Documents/figures/c14_prop_1.png')
#plt.show()