import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import iris
import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.animation as manimation



# def update_quiver(num, Q, X, Y,u_cube,v_cube,level):
#     """updates the horizontal and vertical vector components    """
#     U = u_cube[num][level].data
#     V = v_cube[num][level].data
#     Q.set_UVC(U,V)
#     ttl.set_text(str(0+num))
#     return Q,ttl


# def animate(i, Q, X, Y,u_cube,v_cube,level): 
#     tmp = cube[i].collapsed(['time','air_pressure'],iris.analysis.MEAN)
#     z = tmp.data
#     cont = plt.contourf(X, Y, z, np.linspace(-0.075,0.075,20))
#     """updates the horizontal and vertical vector components    """
#     U = u_cube[i][level].data
#     V = v_cube[i][level].data
#     Q = ax.quiver(X, Y, U,V)
#     #Q.set_UVC(U,V)
#     ttl.set_text(str(0+i))
#     return cont  


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
        comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=1800*4)

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

v_cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/vwnd_monthly_low_res.nc')
u_cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/uwnd_monthly_low_res.nc')

cube = iris.load_cube('/data/NAS-ph290/ph290/reanalysis/NCEP_v2/omega_monthly_low_res.nc')

shape = np.shape(cube)

level = 0
X = cube[0][0].coord('longitude').points
Y = cube[0][0].coord('latitude').points


plt.close('all')
fig, ax = plt.subplots(1,1)    
ax = plt.axes(projection=ccrs.PlateCarree()) 
#ax.set_global()

plot = ax.contourf(X,Y,cube[0][level].data,31)
Q = ax.quiver(X, Y, u_cube[0][level].data, v_cube[0][level].data)
plt.gca().coastlines()
fig.colorbar(plot,orientation = 'horizontal')
ttl = ax.text(.5, 1.005, '0', transform = ax.transAxes)
#anim = animation.FuncAnimation(fig, animate, fargs=(Q, X, Y,u_cube,v_cube,level), frames = 3,interval=2, blit=False)
#shape[0]


with writer.saving(fig, "/home/ph290/Documents/figures/w_velocity_and_windse.mp4", 100):
	for i in range(shape[0]): 
		print i
		tmp = cube[i].collapsed(['time','air_pressure'],iris.analysis.MEAN)
		z = tmp.data
		cont = plt.contourf(X, Y, z, np.linspace(-0.075,0.075,20))
		"""updates the horizontal and vertical vector components    """
		U = u_cube[i][level].data
		V = v_cube[i][level].data
		Q = ax.quiver(X, Y, U,V)
		ttl.set_text(str(0+i))
		writer.grab_frame()


#plt.show()
#anim.save('/home/ph290/Documents/figures/w_velocity_and_windsc.mp4', writer=writer)
#anim.save('/home/ph290/Documents/figures/basic_animation.mp4', fps=8, extra_args=['-vcodec', 'libx264'])


plt.close('all')
fig, ax = plt.subplots(1,1)    
ax = plt.axes(projection=ccrs.PlateCarree()) 
#ax.set_global()

cube_tmp = cube.collapsed(['time','air_pressure'],iris.analysis.MEAN)
u_cube_tmp = u_cube.collapsed(['time'],iris.analysis.MEAN)
v_cube_tmp = v_cube.collapsed(['time'],iris.analysis.MEAN)

plot = ax.contourf(X,Y,cube_tmp.data,31)
Q = ax.quiver(X, Y, u_cube_tmp[0].data, v_cube_tmp[0].data)
plt.gca().coastlines()
fig.colorbar(plot,orientation = 'horizontal')
ttl = ax.text(.5, 1.005, '0', transform = ax.transAxes)
#plt.show()
plt.savefig('/home/ph290/Documents/figures/w_velocity_and_windse.png')
