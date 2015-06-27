import iris
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import iris.analysis.cartography
import iris.plot as iplt
import matplotlib.cm as mpl_cm

cube2 = iris.load_cube('/home/ph290/data1/observations/hadisst/HadISST_sst.nc')
data = cube2[0].data
data2 = data[np.where(data.mask == False)]
total_no_boxes = len(data2)

cube = iris.load_cube('/home/ph290/data1/observations/SOCAT_tracks_gridded_monthly_v2.nc','fCO2 mean - per cruise weighted')

cube.data[np.where(cube.data > 0.0)] = 1.0

coord  = cube.coord('TMNTH')
year = np.array([coord.units.num2date(value).year for value in coord.points])
month = np.array([coord.units.num2date(value).month for value in coord.points])

no_points=[]

for i in np.arange(cube.shape[0]):
    data = cube[i].data
    data2 = data[np.where(data.mask == False)]
    no_points = np.append(no_points,len(data2))

plt.close('all')
plt.plot(year+month/12.0,no_points/total_no_boxes)
plt.savefig('/home/ph290/Documents/figures/no_datapoints1.ps')

brewer_cmap = mpl_cm.get_cmap('brewer_Reds_09')

plt.close('all')
loc = np.where((year <= 2012) & (year > 2002))
cube_collapsed1 = cube[loc].collapsed('TMNTH',iris.analysis.MEAN)
cube_collapsed1.data[np.where(cube_collapsed1.data > 0.0)] = 1.0

plt.close('all')
fig = plt.figure()
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-30))
ax.set_global()
iplt.contourf(cube_collapsed1,np.linspace(0,1.5),cmap=brewer_cmap)
ax.add_feature(cartopy.feature.LAND,facecolor='#f6f6f6')
ax.coastlines(resolution='50m', color='black', linewidth=1)
plt.savefig('/home/ph290/Documents/figures/no_datapoints2.ps')

plt.close('all')
loc = np.where((year <= 1990) & (year > 1980))
cube_collapsed2 = cube[loc].collapsed('TMNTH',iris.analysis.MEAN)
cube_collapsed2.data[np.where(cube_collapsed2.data > 0.0)] = 1.0

plt.close('all')
fig = plt.figure()
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-30))
ax.set_global()
iplt.contourf(cube_collapsed2,np.linspace(0,1.5),cmap=brewer_cmap)
ax.add_feature(cartopy.feature.LAND,facecolor='#f6f6f6')
ax.coastlines(resolution='50m', color='black', linewidth=1)
plt.savefig('/home/ph290/Documents/figures/no_datapoints3.ps')
