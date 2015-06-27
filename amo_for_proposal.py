import iris.analysis
import iris
import cartopy
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.analysis.cartography
import monthly_to_yearly as my
import scipy
from scipy import signal
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.cartography
import iris.analysis.stats

file_name = '/home/ph290/data1/observations/hadisst/HadISST_sst.nc'
cube_in = my.monthly_to_yearly(iris.load_cube(file_name))
cube = cube_in.copy()
cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()

cube2 = cube.copy()

cube_data = cube.data
cube_data_detrended = scipy.signal.detrend(cube_data, axis=0)
cube.data = np.ma.array(cube_data_detrended)
cube.data.mask = cube_in.data.mask


lon_west = -75.0
lon_east = -7.5
lat_south = 0.0
lat_north = 60.0

region = iris.Constraint(longitude=lambda v: lon_west <= v <= lon_east,latitude=lambda v: lat_south <= v <= lat_north)
cube_region = cube.extract(region)
cube2_region = cube2.extract(region)

grid_areas = iris.analysis.cartography.area_weights(cube_region)
amo_regoin_area_avged = cube_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
amo2_regoin_area_avged = cube2_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

data = amo_regoin_area_avged.data[0:-1].data
sorted = np.argsort(data)

high_amo = cube[sorted[0:49]].collapsed('time',iris.analysis.MEAN)
low_amo = cube[sorted[-50:-1]].collapsed('time',iris.analysis.MEAN)

diff = iris.analysis.maths.subtract(low_amo,high_amo)
diff_region = diff.extract(region)

# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# iplt.contourf(diff_region,30)
# #iplt.pcolormesh(diff_region)
# plt.gca().stock_img()
# #plt.gca().coastlines()
# ax.add_feature(cartopy.feature.LAND)
# ax.add_feature(cartopy.feature.OCEAN)
# ax.add_feature(cartopy.feature.COASTLINE)
# #ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
# ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
# ax.add_feature(cartopy.feature.RIVERS)
# plt.show()

coord = amo_regoin_area_avged.coord('time')
dt = coord.units.num2date(coord.points)
year = np.array([coord.units.num2date(value).year for value in coord.points])
year = year[0:-1]

plot_data = amo_regoin_area_avged.data[0:-1]
plot_datab = amo2_regoin_area_avged.data[0:-1]

# fig, ax = plt.subplots()
# ax.plot(year,plot_data)
# plot_data_positive = plot_data.copy()
# plot_data_positive[np.where(plot_data <= 0.0)] = 0.0
# ax.fill_between(year,plot_data_positive, 0.0, facecolor='red', alpha=0.65)
# plot_data_negative = plot_data.copy()
# plot_data_negative[np.where(plot_data >= 0.0)] = 0.0
# ax.fill_between(year,plot_data_negative, 0.0, facecolor='blue', alpha=0.65)
# plt.xlabel('year')
# plt.ylabel('AMO box SST anomaly ($^{\circ}\mathrm{C}$)')
# plt.show()


'''
global surface T
'''

file_name2 = '/home/ph290/data1/observations/hadcrut4/HadCRUT.4.2.0.0.median.nc'
hadcru_cube = my.monthly_to_yearly(iris.load_cube(file_name2,'near_surface_temperature_anomaly'))
hadcru_cube = hadcru_cube[20:-1]
hadcru_cube.coord('latitude').guess_bounds()
hadcru_cube.coord('longitude').guess_bounds()
hadcru_cube_detrended = hadcru_cube.copy()
hadcru_cube_detrended.data = scipy.signal.detrend(hadcru_cube_detrended.data, axis=0)

coord = hadcru_cube.coord('time')
dt = coord.units.num2date(coord.points)
hadcru_year = np.array([coord.units.num2date(value).year for value in coord.points])

amo_cube = hadcru_cube_detrended.copy()
amo_cube = iris.analysis.maths.multiply(amo_cube, 0.0)

plot_data2 = np.swapaxes(np.swapaxes(np.tile(plot_data.copy(),[36,72,1]),1,2),0,1)
amo_cube = iris.analysis.maths.add(amo_cube, plot_data2)

out_cube = iris.analysis.stats.pearsonr(hadcru_cube_detrended, amo_cube, corr_coords=['time'])


hadcru_cube_ts = hadcru_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN)

plt.plot(hadcru_year, hadcru_cube_ts.data-np.mean(hadcru_cube_ts.data))
plt.plot(year,plot_data)
plt.show()

hadcru_cube_amo = hadcru_cube_detrended.copy()

hadcru_cube_amo.data[:,np.where(hadcru_cube[0].coord('latitude').points > 60),:] = 0.0
hadcru_cube_amo.data[:,np.where(hadcru_cube[0].coord('latitude').points < 0),:] = 0.0
hadcru_cube_amo.data[:,:,np.where((hadcru_cube[0].coord('longitude').points > -7.5) & (hadcru_cube[0].coord('longitude').points < 180))] = 0.0
hadcru_cube_amo.data[:,:,np.where((hadcru_cube[0].coord('longitude').points < -75))] = 0.0

hadcru_cube_amo_ts = hadcru_cube_amo.collapsed(['latitude','longitude'],iris.analysis.MEAN)

plt.plot(hadcru_year, hadcru_cube_ts.data-np.mean(hadcru_cube_ts.data))
plt.plot(hadcru_year, hadcru_cube_amo_ts.data-np.mean(hadcru_cube_amo_ts.data))
plt.show()

test = iris.analysis.maths.multiply(amo_cube,out_cube)
x = test.collapsed(['latitude','longitude'],iris.analysis.MEAN)

# plt.plot(hadcru_year, hadcru_cube_ts.data-np.mean(hadcru_cube_ts.data))
# plt.plot(year,x.data)
# plt.plot(year,plot_data)
# plt.show()

fig, ax = plt.subplots()
plot_data = amo_regoin_area_avged.data[0:-1]
p1, = ax.plot(year,plot_data,'k',linewidth = 3)
plot_data_positive = plot_data.copy()
plot_data_positive[np.where(plot_data <= 0.0)] = 0.0
ax.fill_between(year,plot_data_positive, 0.0, facecolor='red', alpha=0.65)
plot_data_negative = plot_data.copy()
plot_data_negative[np.where(plot_data >= 0.0)] = 0.0
ax.fill_between(year,plot_data_negative, 0.0, facecolor='blue', alpha=0.65)
p2, = plt.plot(hadcru_year, hadcru_cube_ts.data-np.mean(hadcru_cube_ts.data[0:10]),'g',linewidth = 3)
p3, = ax.plot(year,plot_datab - np.mean(plot_datab[0:10]),'k--',linewidth = 3)
# plt.plot(year,x.data)
plt.xlabel('year')
plt.ylabel('temperature anomaly (K)')
plt.legend([p1, p3, p2], ["AMO box mean SST anomaly (detrended)","AMO box mean SST (not detrended)", "global surface temperature anomaly"],loc='upper left').draw_frame(False)
plt.show()



