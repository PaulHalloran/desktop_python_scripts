import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import iris
from matplotlib import animation

def update_quiver(num, Q, X, Y,u_cube,v_cube,level):
    """updates the horizontal and vertical vector components    """
    U = u_cube[num][level].data
    V = v_cube[num][level].data
    Q.set_UVC(U,V)
    ttl.set_text(str(0+num))
    return Q,ttl


Writer = animation.writers['ffmpeg']
writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800*4)

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
shape = np.shape(v_cube)

level = 0



X = u_cube[0][0].coord('longitude').points
Y = u_cube[0][0].coord('latitude').points  

plt.close('all')
fig, ax = plt.subplots(1,1)    
ax = plt.axes(projection=ccrs.PlateCarree())  
Q = ax.quiver(X, Y, u_cube[0][level].data, v_cube[0][level].data)
plt.gca().coastlines()
ttl = ax.text(.5, 1.005, '0', transform = ax.transAxes)
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y,u_cube,v_cube,level),frames = shape[0],
                               interval=2, blit=False)
#plt.show()
anim.save('/home/ph290/Documents/figures/winds_monthly.mp4', writer=writer)
