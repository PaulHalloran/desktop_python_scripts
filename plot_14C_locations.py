import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import iris
import iris.plot as iplt
import iris.quickplot as qplt

#latitude = iris.coords.DimCoord(np.linspace(-90,90,180.0*4.0), standard_name='latitude', units='degrees')
#longitude = iris.coords.DimCoord(np.linspace(0,360,360.0*4.0), standard_name='longitude', units='degrees')
#regridding_cube = iris.cube.Cube(np.zeros((180.0*4.0,360.0*4.0), np.float32),standard_name='sea_surface_temperature', long_name='Sea Surface Temperature', var_name='tos', units='K',dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
#bathy_cube = iris.load_cube('/home/ph290/data0/misc_data/ETOPO1_Bed_c_gmt4.grd')
#bathy_cube_regrid = bathy_cube.regrid(regridding_cube, iris.analysis.Linear())



c14_timeseries = {}
c14_timeseries['ref'] = ['Druffel, 1989','Druffel and Suess, 1983','Nozakui et al., 1978','Druffel, 1996','Toggweiler et al., 1991','Grumet et al., 2002','Grumet et al., 2004','Shen et al., 2004','Konishi et al., 1981','Konishi et al., 1981','Asami et al., 2005','Druffel and Griffin, 1999','Druffel and Griffin, 1999','Druffel, 1987','Druffel, 1987','Druffel et al., 2001','Guilderson et al., 2000','Toggweiler et al., 1991','Schmidt et al., 2004','Guilderson et al., 2004','Grottoli et al., 2003','Druffel, 1987','Guilderson and Schrag, 1998','Druffel, 1987','Druffel, 1987','Guilderson and Schrag, 1998','Weidman 1995','Weidman 1995','Witbaard et al., 1994','Weidman 1995','Scourse at al., 2012','Wideman and Jones, 1996','Kilda et al., 2007']
c14_timeseries['creature'] = ['coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','coral','bivalve','bivalve','bivalve','bivalve','bivalve','bivalve','bivalve']
c14_timeseries['lat'] = np.array([24.57,15.0,32.0,-17.3,12.0,-3.32,-0.08,22.33,26.38,26.14,13.35,-23.0,-22.06,23.43,21.0,21.18,-21.14,-18.0,-10.0,-9.0,3.54,3.52,-0.5,7.48,-0.5,-0.5,70.0,54.0,54.0,66.0,66.0,41.0,44.0])
c14_timeseries['lon'] = np.array([-80.33,-85.0,-64.0,-39.2,42.0,39.52,98.31,114.32,127.52,127.41,144.50,152.0,153.0,-166.06,-158.0,-158.07,-159.49,179.0,161.0,160.0,-159.19,-159.23,166.0,-81.45,-90.0,-90.0,19.0,7.0,4.0,-19.0,-18.0,-67.0,-61.0])
c14_timeseries['start_date'] = np.array([1900,1850,1770,1950,1951,1947,1944,1977,1940,1912,1787,1849,1635,1957,1958,1890,1950,1925,1944,1942,1922,1949,1947,1950,1930,1956,1940,1948,1927,1874.0,1925,1939,1963])
c14_timeseries['end_date'] = np.array([1952,1983,1976,1982,1985,1998,1992,1998,1980,1979,2000,1983,1991,1978,1979,1979,1997,1978,1994,1995,1956,1979,1995,1980,1977,1983,1992,1989,1986,1991,2005,1989,1978])

#rotated_pole = ccrs.RotatedPole(pole_latitude=45, pole_longitude=180)
projection = ccrs.PlateCarree()
#projection = ccrs.NorthPolarStereo()

cm1 = plt.cm.get_cmap('Greys')

x = c14_timeseries['lon']
y = c14_timeseries['lat']
z = c14_timeseries['end_date'] - c14_timeseries['start_date']

#ax = plt.subplot(111, projection=projection)
ax = plt.subplot(111, projection=projection)
#iplt.contourf(bathy_cube_regrid,np.linspace(-6000,0,31),cmap=cm1)
#ax.imshow(np.roll(np.roll(bathy_cube_regrid.data,90+25*2,axis=0),180,axis = 1), transform=ccrs.PlateCarree())
#img_extent = (-350*4,00,-180*4/2,90)
img_extent = (-160*2.0-40,20*2.0-40,-90,90)
ax.imshow(bathy_cube_regrid.data, extent=img_extent, transform=ccrs.PlateCarree(),cmap=cm1)
img_extent = (-160*2.0-40+360,20*2.0-40+360,-90,90)
ax.imshow(bathy_cube_regrid.data, extent=img_extent, transform=ccrs.PlateCarree(),cmap=cm1)

#plt.show()

#iplt.contourf(new_cube_region_II,np.linspace(-6000,0,31),cmap=cm1)
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, alpha=0.1)
#ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
#ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
#ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
#ax.add_feature(cartopy.feature.RIVERS)

cm = plt.cm.get_cmap('RdYlBu')

for i in enumerate(range(len(c14_timeseries['ref']))):
    i = i[0]
    x = c14_timeseries['lon'][i]
    y = c14_timeseries['lat'][i]
    z = c14_timeseries['end_date'][i] - c14_timeseries['start_date'][i]
    if c14_timeseries['creature'][i] == 'coral':
        sc = ax.scatter(x, y, c=z, vmin=0, vmax=200, marker='o', s=300, transform=projection)
    if c14_timeseries['creature'][i] == 'bivalve':
        sc1 = ax.scatter(x, y, c=z, vmin=0, vmax=200, marker='*', s=300, transform=projection)

bar = plt.colorbar(sc)
bar.set_label('length of $^{14}$C record (years)')
ax.gridlines()
ax.set_extent([-100, 20, 0, 75])

plt.legend((sc,sc1),
           ('coral', 'bivalve'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=8)


plt.savefig('/home/ph290/Documents/figures/c14_locations_stuff.pdf')
#plt.show()



