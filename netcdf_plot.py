from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

#file='/project/champ/data/cmip5/output1/NOAA-GFDL/GFDL-ESM2M/rcp85/mon/atmos/Amon/r1i1p1/v20111228/tas/tas_Amon_GFDL-ESM2M_rcp85_r1i1p1_202601-203012.nc'
file='/project/champ/data/cmip5/output1/NOAA-GFDL/GFDL-ESM2M/historical/mon/ocnBgchem/Omon/r1i1p1/v20110601/fgco2/fgco2_Omon_GFDL-ESM2M_historical_r1i1p1_199601-200012.nc'
nc = NetCDFFile(file)

    # data from http://water.weather.gov/precip/
tas = nc.variables['fgco2']
tas2=tas[0]
data = tas2[:]
latcorners = nc.variables['lat'][:]
loncorners = -nc.variables['lon'][:]
    #lon_0 = -nc.variables['true_lon'].getValue()
    #lat_0 = nc.variables['true_lat'].getValue()
    # create figure and axes instances
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # create polar stereographic Basemap instance.

m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='gray',lake_color='white')
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='aqua')

ny = data.shape[0]; nx = data.shape[1]
lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
x, y = m(lons, lats) # compute map proj coordinates.

clevs = np.linspace(data.min(),data.max(),20.0)
cs = m.contourf(x,y,data,clevs,cmap=plt.hot())
cbar = m.colorbar(cs,location='bottom',pad="5%")
#cbar.set_label('mm')
# add title
#plt.title(prcpvar.long_name+' for period ending '+prcpvar.dateofdata)

plt.show()
