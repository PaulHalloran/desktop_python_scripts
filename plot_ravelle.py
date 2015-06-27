import matplotlib.pyplot as plt
import iris
import numpy as np
import carbchem_cube_revelle
import iris.quickplot as qplt
import iris
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import iris.coord_categorisation
import iris.analysis.cartography
import iris.analysis
import scipy
from scipy import signal
import numpy as np
import cartopy

'''

directory = '/home/ph290/Documents/teaching/'

t_file = 'temperature_annual_1deg.nc'
s_file = 'salinity_annual_1deg.nc'
tco2_file = 'Total_CO2.nc'
alk_file = 'Alk.nc'

alk = iris.load_cube(directory+alk_file,'Alkalinity')/1.0e3
t =  iris.load_cube(directory+t_file,'Statistical Mean').collapsed('time',iris.analysis.MEAN)+273.15
s =  iris.load_cube(directory+s_file,'Statistical Mean').collapsed('time',iris.analysis.MEAN)
tco2 =  iris.load_cube(directory+tco2_file)/1.0e3

alk = alk[0]
t = t[0]
s = s[0]
tco2 = tco2[0]

revelle_factor = carbchem_cube_revelle.carbchem_revelle(10,t.data.fill_value,t,s,tco2,alk)

'''

plt_cube = revelle_factor
lats = plt_cube.coord('latitude').points
lons = plt_cube.coord('longitude').points
lons, lats = np.meshgrid(lons, lats)

x1 = [-70,-10,-10,-70,-70]
y1 = [50,50,70,70,50]

x2 = [-70,-10,-10,-70,-70]
y2 = [0,0,30,30,0]

#box model
x3 = [-70,10,10,-70,-70]
y3 = [48,48,90,90,48]

plt.close('all')
plt.figure()
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-40, central_latitude=45))
ax.set_global()
CS = ax.contourf(lons,lats,plt_cube.data,np.linspace(8,14,51),transform = ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)
ax.coastlines()
ax.plot(x1, y1, marker='o',color='b', transform = ccrs.PlateCarree())
ax.fill(x1, y1, color='coral', transform= ccrs.PlateCarree(), alpha=0.4)
ax.plot(x2, y2, marker='o',color='r', transform = ccrs.PlateCarree())
ax.fill(x2, y2, color='coral', transform= ccrs.PlateCarree(), alpha=0.4)

#ax.plot(x3, y3, marker='o',color='k', transform = ccrs.PlateCarree())
#ax.fill(x3, y3, color='k', transform= ccrs.PlateCarree(), alpha=0.4)




cbar = plt.colorbar(CS)
cbar.ax.set_xlabel('Revelle Factor')
plt.show(block = False)
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/revelle.png')

