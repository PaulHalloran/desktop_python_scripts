import iris
import numpy as np
import glob
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

files = glob.glob('/home/ph290/Documents/teaching/ssh/ssh2/dh*')
files.sort()

data = np.zeros((12,180,360))
for i,file in enumerate(files):
	tmp_data = np.genfromtxt(file)
	tmp_data2 = np.reshape(tmp_data,[180,360])
	data[i,:,:] = tmp_data2

data_mean = np.mean(data,axis = 0)

spacing_deg = 1
lat_array = range(-90, 90, spacing_deg)
lon_array = range(0, 360, spacing_deg)
latitude = iris.coords.DimCoord(lat_array, standard_name='latitude',
                    units='degrees')
longitude = iris.coords.DimCoord(lon_array, standard_name='longitude',
                     units='degrees')
cube = iris.cube.Cube(np.zeros((np.ceil(180.0/spacing_deg), 360.0/spacing_deg), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

for i in enumerate(data_mean[:,0]):
	for j in enumerate(data_mean[0,:]):
		if data_mean[i[0],j[0]] <= -90.0:
			data_mean[i[0],j[0]] = np.nan
    
data_mean = np.ma.masked_where(data_mean == np.nan,data_mean)
data_mean = np.ma.masked_where(data_mean <= -90.0,data_mean)
cube.data = data_mean

iris.fileformats.netcdf.save(cube,'/home/ph290/Documents/teaching/sea_surface_height_woa.nc')

lats = cube.coord('latitude').points
lons = cube.coord('longitude').points
lons, lats = np.meshgrid(lons, lats)

plt.figure(figsize=(5, 5))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0.0, central_latitude=-90.0, globe=None))
ax.set_global()
cb = ax.contourf(lons,lats,cube.data,np.linspace(-50,250,51),transform=ccrs.PlateCarree())
ax.coastlines(resolution='110m')
cb1 = plt.colorbar(cb, orientation="vertical")
cb1.set_label('Sea surface height anomaly (cm)')
# plt.show()
plt.savefig('/home/ph290/Documents/figures/sea_surface_height_antarctic.pdf', dpi=300)

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# import numpy as np
# from matplotlib import cm
# from matplotlib import pyplot as plt
# 
# step = 0.04
# maxval = 1.0
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# ax.plot_surface(lons,lats,data_mean, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0)
# plt.show()