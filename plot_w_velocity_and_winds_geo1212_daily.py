import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import iris
import matplotlib



v_cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/vwnd.nc')
u_cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/uwnd.nc')

cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/omega.nc')

shape = np.shape(cube)

level = 0
X = cube[0][0].coord('longitude').points
Y = cube[0][0].coord('latitude').points


plt.close('all')
fig, ax = plt.subplots(1,1)    
ax = plt.axes(projection=ccrs.PlateCarree()) 
#ax.set_global()

i=-200

plot = ax.contourf(X,Y,cube[i][level].data,np.linspace(-0.3,0.3,31))
Q = ax.quiver(X, Y, u_cube[i][level].data, v_cube[0][level].data)
plt.gca().coastlines()
fig.colorbar(plot,orientation = 'horizontal')
ttl = ax.text(.5, 1.005, '0', transform = ax.transAxes)
#anim = animation.FuncAnimation(fig, animate, fargs=(Q, X, Y,u_cube,v_cube,level), frames = 3,interval=2, blit=False)
#shape[0]


#plt.show()
plt.savefig('/home/ph290/Documents/figures/w_velocity_and_winds_4.png')
#anim.save('/home/ph290/Documents/figures/basic_animation.mp4', fps=8, extra_args=['-vcodec', 'libx264'])

