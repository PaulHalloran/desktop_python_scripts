import iris
import matplotlib.pyplot as plt
import numpy as np
import iris.quickplot as qplt


v_cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/vwnd_monthly.nc')
w_cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/omega_monthly.nc')

v_cube.coord('latitude').guess_bounds()
v_cube.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(v_cube)

v_cube_long = v_cube.collapsed(['time','longitude'],iris.analysis.MEAN, weights=grid_areas)

w_cube_long = w_cube.collapsed(['time','longitude'],iris.analysis.MEAN, weights=grid_areas)

ps = v_cube_long.coord('air_pressure').points

height_m = (np.power((ps*100.0)/101325.0,(1/5.25588))-1)/(-2.25577e-5)

#qplt.contourf(v_cube_time[0],31)
#plt.gca().coastlines()
#plt.show()

plt.figure(num=None, figsize=(8, 4), dpi=300)
plt.contourf(v_cube_long.coord('latitude').points,height_m,v_cube_long.data,np.linspace(-2,2,31))
plt.contour(w_cube_long.coord('latitude').points,height_m,w_cube_long.data,np.linspace(-0.05,0.051,21),colors='k')
plt.xlabel('latitude (degrees)')
plt.ylabel('height (m)')
plt.ylim([0,20000])
plt.tight_layout()
plt.savefig('/home/ph290/Documents/figures/w_v_velocities.png')
#plt.show()

#plt.contourf(w_cube_long.coord('latitude').points,height_m,w_cube_long.data,np.linspace(-0.05,0.051,31))
#plt.xlabel('latitude (degrees)')
#plt.ylabel('height (m)')
#plt.savefig('/home/ph290/Documents/figures/w_velocities.png')
#plt.show()


#plt.show()



