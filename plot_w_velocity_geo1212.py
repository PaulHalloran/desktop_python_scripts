import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import iris
from matplotlib import animation


def animate(i): 
    tmp = cube[i].collapsed(['time','air_pressure'],iris.analysis.MEAN)
    z = tmp.data
    cont = plt.contourf(X, Y, z, np.linspace(-0.075,0.075,20))
    ttl.set_text(str(0+i))
    return cont  




Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800*4)

#v_cube.coord('air_pressure').points/10.0
#array([ 100. ,   92.5,   85. ,   70. ,   60. ,   50. ,   40. ,   30. ,
#         25. ,   20. ,   15. ,   10. ,    7. ,    5. ,    3. ,    2. ,
#          1. ], dtype=float32)
#0 = 0km
#1 = 750km
#2 = 1.8km
#3 = 3km
#4 = 4km
#5 = 5.6km
#6 = 7km


cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/omega_monthly.nc')

shape = np.shape(cube)

level = 0
X = cube[0][0].coord('longitude').points
Y = cube[0][0].coord('latitude').points


plt.close('all')
fig, ax = plt.subplots(1,1)    
ax = plt.axes(projection=ccrs.PlateCarree()) 
ax.set_global()
Q = ax.contourf(X,Y,cube[0][level].data,31)
plt.gca().coastlines()
ttl = ax.text(.5, 1.005, '0', transform = ax.transAxes)
anim = animation.FuncAnimation(fig, animate,frames = shape[0],interval=2, blit=False)


#plt.show()
anim.save('/home/ph290/Documents/figures/w_velocity_levels_averaged.mp4', writer=writer)
