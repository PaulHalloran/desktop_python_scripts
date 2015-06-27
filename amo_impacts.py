import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy.ma as ma
import running_mean
from scipy import signal
import matplotlib.cm as mpl_cm

'''
Firstly working out which ar ethe high and low AMO years
'''

hadisst_file = '/home/ph290/data1/observations/hadisst/HadISST_sst.nc'

hadisst = iris.load_cube(hadisst_file)

iris.coord_categorisation.add_year(hadisst, 'time', name='year')
hadisst_annual = hadisst.aggregated_by('year', iris.analysis.MEAN)

lon_west = -75.0
lon_east = -7.5
lat_south = 0.0
lat_north = 60.0

region = iris.Constraint(longitude=lambda v: lon_west <= v <= lon_east,latitude=lambda v: lat_south <= v <= lat_north)
hadisst_annual_region = hadisst_annual.extract(region)


hadisst_annual_region.coord('latitude').guess_bounds()
hadisst_annual_region.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(hadisst_annual_region)
hadisst_annual_region_ts = hadisst_annual_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

coord = hadisst_annual.coord('time')
dt = coord.units.num2date(coord.points)
hadisst_year = np.array([coord.units.num2date(value).year for value in coord.points])

hadisst_annual_region_ts_detrend = signal.detrend(hadisst_annual_region_ts.data)

hadisst_year = hadisst_year[0:-1]
hadisst_annual_region_ts_detrend = hadisst_annual_region_ts_detrend[0:-1]

high_amo_years = hadisst_year[np.where(hadisst_annual_region_ts_detrend > np.mean(hadisst_annual_region_ts_detrend))]
low_amo_years = hadisst_year[np.where(hadisst_annual_region_ts_detrend < np.mean(hadisst_annual_region_ts_detrend))]

'''
Next reading in 20CR T and precip
'''

directory = '/data/temp/ph290/reanalysis_20cr/'

twenty_cr_t = iris.load_cube(directory+'air.2m.mon.mean.nc','air_temperature')
twenty_cr_p = iris.load_cube(directory+'prate.mon.mean.nc','precipitation_flux')

'''
processing
'''

#iris.coord_categorisation.add_season(twenty_cr_t, 'time', name='season')
iris.coord_categorisation.add_season_year(twenty_cr_t, 'time', name='season_year')
iris.coord_categorisation.add_season_number(twenty_cr_t, 'time', name='season_number')
twenty_cr_t_seasonal = twenty_cr_t.aggregated_by(['season_year','season_number'], iris.analysis.MEAN)

iris.coord_categorisation.add_season_year(twenty_cr_p, 'time', name='season_year')
iris.coord_categorisation.add_season_number(twenty_cr_p, 'time', name='season_number')
twenty_cr_p_seasonal = twenty_cr_p.aggregated_by(['season_year','season_number'], iris.analysis.MEAN)

twenty_cr_t_seasonal_detrend  = twenty_cr_t_seasonal.copy()
twenty_cr_t_seasonal_detrend.data = signal.detrend(twenty_cr_t_seasonal.data,axis = 0)

twenty_cr_p_seasonal_detrend  = twenty_cr_p_seasonal.copy()
twenty_cr_p_seasonal_detrend.data = signal.detrend(twenty_cr_p_seasonal.data,axis = 0)

coord = twenty_cr_t_seasonal.coord('time')
dt = coord.units.num2date(coord.points)
twenty_cr_years = np.array([coord.units.num2date(value).year for value in coord.points])

high_amo_indicies = np.in1d(twenty_cr_years, high_amo_years)
low_amo_indicies = np.in1d(twenty_cr_years, low_amo_years)
#in_both = np.intersect1d(twenty_cr_years, high_amo_years)

high_amo_t = twenty_cr_t_seasonal_detrend[high_amo_indicies]
high_amo_p = twenty_cr_p_seasonal_detrend[high_amo_indicies]

low_amo_t = twenty_cr_t_seasonal_detrend[low_amo_indicies]
low_amo_p = twenty_cr_p_seasonal_detrend[low_amo_indicies]

#djf', 'mam', 'jja', 'son' = 0,1,2,3
high_amo_t_summer = high_amo_t.extract(iris.Constraint(season_number = 2)).collapsed('time',iris.analysis.MEAN)
high_amo_t_winter = high_amo_t.extract(iris.Constraint(season_number = 0)).collapsed('time',iris.analysis.MEAN)

low_amo_t_summer = low_amo_t.extract(iris.Constraint(season_number = 2)).collapsed('time',iris.analysis.MEAN)
low_amo_t_winter = low_amo_t.extract(iris.Constraint(season_number = 0)).collapsed('time',iris.analysis.MEAN)

high_amo_p_summer = high_amo_p.extract(iris.Constraint(season_number = 2)).collapsed('time',iris.analysis.MEAN)
high_amo_p_winter = high_amo_p.extract(iris.Constraint(season_number = 0)).collapsed('time',iris.analysis.MEAN)

low_amo_p_summer = low_amo_p.extract(iris.Constraint(season_number = 2)).collapsed('time',iris.analysis.MEAN)
low_amo_p_winter = low_amo_p.extract(iris.Constraint(season_number = 0)).collapsed('time',iris.analysis.MEAN)

summer_t_range = np.linspace(-1.2,1.2,50)
winter_t_range = np.linspace(-2.0,2.0,50)

plt.close('all')
cmap = mpl_cm.get_cmap('bwr')
plt.figure()
qplt.contourf(high_amo_t_summer-low_amo_t_summer,30,cmap=cmap,levels = summer_t_range)
plt.gca().coastlines()
plt.title('20cr summer dT high minus low')
plt.savefig('/home/ph290/Documents/figures/AMO_impacts_1_20cr.png')
#plt.show()

plt.figure()
qplt.contourf(high_amo_t_winter-low_amo_t_winter,30,cmap=cmap,levels = winter_t_range)
plt.gca().coastlines()
plt.title('20cr winter dT high minus low')
plt.savefig('/home/ph290/Documents/figures/AMO_impacts_2_20cr.png')
#plt.show()

plt.figure()
qplt.contourf(high_amo_p_summer-low_amo_p_summer,30,cmap=cmap,levels = np.linspace(-0.0000225,0.0000225,50))
plt.gca().coastlines()
plt.title('20cr summer dPrecip high minus low')
plt.savefig('/home/ph290/Documents/figures/AMO_impacts_3_20cr.png')
#plt.show()

plt.figure()
qplt.contourf(high_amo_p_winter-low_amo_p_winter,30,cmap=cmap,levels = np.linspace(-0.0000225,0.0000225,50))
plt.gca().coastlines()
plt.title('20cr winter dPrecip high minus low')
plt.savefig('/home/ph290/Documents/figures/AMO_impacts_4_20cr.png')
#plt.show()


'''
Next reading in ERA-clim T and precip
'''

directory2 = '/data/temp/ph290/era-clim/'

cube1t = iris.load_cube(directory2+'t_p_ens1.nc','2 metre temperature')
cube2t = iris.load_cube(directory2+'t_p_ens2.nc','2 metre temperature')
cube3t = iris.load_cube(directory2+'t_p_ens3.nc','2 metre temperature')
cube4t = iris.load_cube(directory2+'t_p_ens4.nc','2 metre temperature')
cube5t = iris.load_cube(directory2+'t_p_ens5.nc','2 metre temperature')
cube6t = iris.load_cube(directory2+'t_p_ens6.nc','2 metre temperature')
cube7t = iris.load_cube(directory2+'t_p_ens7.nc','2 metre temperature')
cube8t = iris.load_cube(directory2+'t_p_ens8.nc','2 metre temperature')
cube9t = iris.load_cube(directory2+'t_p_ens9.nc','2 metre temperature')
cube10t = iris.load_cube(directory2+'t_p_ens10.nc','2 metre temperature')

cube1p = iris.load_cube(directory2+'t_p_ens1.nc','Total precipitation')
cube2p = iris.load_cube(directory2+'t_p_ens2.nc','Total precipitation')
cube3p = iris.load_cube(directory2+'t_p_ens3.nc','Total precipitation')
cube4p = iris.load_cube(directory2+'t_p_ens4.nc','Total precipitation')
cube5p = iris.load_cube(directory2+'t_p_ens5.nc','Total precipitation')
cube6p = iris.load_cube(directory2+'t_p_ens6.nc','Total precipitation')
cube7p = iris.load_cube(directory2+'t_p_ens7.nc','Total precipitation')
cube8p = iris.load_cube(directory2+'t_p_ens8.nc','Total precipitation')
cube9p = iris.load_cube(directory2+'t_p_ens9.nc','Total precipitation')
cube10p = iris.load_cube(directory2+'t_p_ens10.nc','Total precipitation')


twenty_cr_t = cube1t.copy()
twenty_cr_p = cube1p.copy()

'''
Calculating mean of ERA-clim ensemble
'''

twenty_cr_t.data = np.mean([cube1t.data,cube2t.data,cube3t.data,cube4t.data,cube5t.data,cube6t.data,cube7t.data,cube8t.data,cube9t.data,cube10t.data],axis = 0)

twenty_cr_p.data = np.mean([cube1p.data,cube2p.data,cube3p.data,cube4p.data,cube5p.data,cube6p.data,cube7p.data,cube8p.data,cube9p.data,cube10p.data],axis = 0)



'''
processing
'''

#iris.coord_categorisation.add_season(twenty_cr_t, 'time', name='season')
iris.coord_categorisation.add_season_year(twenty_cr_t, 'time', name='season_year')
iris.coord_categorisation.add_season_number(twenty_cr_t, 'time', name='season_number')
twenty_cr_t_seasonal = twenty_cr_t.aggregated_by(['season_year','season_number'], iris.analysis.MEAN)

iris.coord_categorisation.add_season_year(twenty_cr_p, 'time', name='season_year')
iris.coord_categorisation.add_season_number(twenty_cr_p, 'time', name='season_number')
twenty_cr_p_seasonal = twenty_cr_p.aggregated_by(['season_year','season_number'], iris.analysis.MEAN)

twenty_cr_t_seasonal_detrend  = twenty_cr_t_seasonal.copy()
twenty_cr_t_seasonal_detrend.data = signal.detrend(twenty_cr_t_seasonal.data,axis = 0)

twenty_cr_p_seasonal_detrend  = twenty_cr_p_seasonal.copy()
twenty_cr_p_seasonal_detrend.data = signal.detrend(twenty_cr_p_seasonal.data,axis = 0)

coord = twenty_cr_t_seasonal.coord('time')
dt = coord.units.num2date(coord.points)
twenty_cr_years = np.array([coord.units.num2date(value).year for value in coord.points])

high_amo_indicies = np.in1d(twenty_cr_years, high_amo_years)
low_amo_indicies = np.in1d(twenty_cr_years, low_amo_years)
#in_both = np.intersect1d(twenty_cr_years, high_amo_years)

high_amo_t = twenty_cr_t_seasonal_detrend[high_amo_indicies]
high_amo_p = twenty_cr_p_seasonal_detrend[high_amo_indicies]

low_amo_t = twenty_cr_t_seasonal_detrend[low_amo_indicies]
low_amo_p = twenty_cr_p_seasonal_detrend[low_amo_indicies]

#djf', 'mam', 'jja', 'son' = 0,1,2,3
high_amo_t_summer = high_amo_t.extract(iris.Constraint(season_number = 2)).collapsed('time',iris.analysis.MEAN)
high_amo_t_winter = high_amo_t.extract(iris.Constraint(season_number = 0)).collapsed('time',iris.analysis.MEAN)

low_amo_t_summer = low_amo_t.extract(iris.Constraint(season_number = 2)).collapsed('time',iris.analysis.MEAN)
low_amo_t_winter = low_amo_t.extract(iris.Constraint(season_number = 0)).collapsed('time',iris.analysis.MEAN)

high_amo_p_summer = high_amo_p.extract(iris.Constraint(season_number = 2)).collapsed('time',iris.analysis.MEAN)
high_amo_p_winter = high_amo_p.extract(iris.Constraint(season_number = 0)).collapsed('time',iris.analysis.MEAN)

low_amo_p_summer = low_amo_p.extract(iris.Constraint(season_number = 2)).collapsed('time',iris.analysis.MEAN)
low_amo_p_winter = low_amo_p.extract(iris.Constraint(season_number = 0)).collapsed('time',iris.analysis.MEAN)

cmap = mpl_cm.get_cmap('bwr')
plt.figure()
qplt.contourf(high_amo_t_summer-low_amo_t_summer,30,cmap=cmap,levels = summer_t_range)
plt.gca().coastlines()
plt.title('era-clim summer dT high minus low')
plt.savefig('/home/ph290/Documents/figures/AMO_impacts_1_era_clim.png')
#plt.show()

plt.figure()
qplt.contourf(high_amo_t_winter-low_amo_t_winter,30,cmap=cmap,levels = winter_t_range)
plt.gca().coastlines()
plt.title('era-clim winter dT high minus low')
plt.savefig('/home/ph290/Documents/figures/AMO_impacts_2_era_clim.png')
#plt.show()

plt.figure()
qplt.contourf(high_amo_p_summer-low_amo_p_summer,30,cmap=cmap,levels = np.linspace(-0.002,0.002,50))
plt.gca().coastlines()
plt.title('era-clim summer dPrecip high minus low')
plt.savefig('/home/ph290/Documents/figures/AMO_impacts_3_era_clim.png')
#plt.show()

plt.figure()
qplt.contourf(high_amo_p_winter-low_amo_p_winter,30,cmap=cmap,levels = np.linspace(-0.002,0.002,50))
plt.gca().coastlines()
plt.title('era-clim winter dPrecip high minus low')
plt.savefig('/home/ph290/Documents/figures/AMO_impacts_4_era_clim.png')
#plt.show()
