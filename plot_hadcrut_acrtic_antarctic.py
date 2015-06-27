import numpy as np
import matplotlib.pyplot as plt
import iris
import iris.analysis
import iris.quickplot as qplt
import iris.analysis.cartography
import cartopy.crs as ccrs


t_file = '/home/ph290/data1/observations/hadcrut4/HadCRUT.4.2.0.0.median.nc'

t = iris.load_cube(t_file,'near_surface_temperature_anomaly')
t_mean = t[-12*20:-1].collapsed('time, iris.analysis.MEAN)



t_annual.coord('latitude').guess_bounds()
t_annual.coord('longitude').guess_bounds()

