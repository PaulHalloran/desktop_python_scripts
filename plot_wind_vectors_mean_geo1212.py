import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import iris
from matplotlib import animation




v_cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/vwnd_monthly_low_res.nc')
u_cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/uwnd_monthly_low_res.nc')
shape = np.shape(v_cube)

level = 0



X = u_cube[0][0].coord('longitude').points
Y = u_cube[0][0].coord('latitude').points  

u_cube_tmp = u_cube.collapsed('time',iris.analysis.MEAN)
v_cube_tmp = v_cube.collapsed('time',iris.analysis.MEAN)

plt.close('all')
fig, ax = plt.subplots(1,1)    
ax = plt.axes(projection=ccrs.PlateCarree())  
Q = ax.quiver(X, Y, u_cube_tmp[level].data, v_cube_tmp[level].data)
plt.gca().coastlines()
# plt.show()
plt.savefig('/home/ph290/Documents/figures/winds_monthly_mean.png')

#anim.save('/home/ph290/Documents/figures/winds_monthly.mp4', writer=writer)
