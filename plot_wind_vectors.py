import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs

def plot_wind_vectors(cube_u_mean,cube_v_mean,level):
	x = cube_u_mean.coord('longitude').points
	y = cube_u_mean.coord('latitude').points                                
	ax = plt.axes(projection=ccrs.PlateCarree())  
	ax.quiver(x, y, cube_u_mean[level].data, cube_v_mean[level].data)
	plt.gca().coastlines()
	plt.show(block = False)

