# import iris
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
# import iris.coord_categorisation
# import iris.analysis.cartography
# import iris.analysis
# import scipy
# from scipy import signal
# import numpy as np
# import cartopy
# import numpy.ma as ma
# import iris.quickplot as qplt
# import glob

# file = '/media/usb_external1/cmip5/temp.nc'
# cube = iris.load_cube(file)
# cube = cube[0:250*12,:,:]
# iris.coord_categorisation.add_month(cube, 'time', name='month')
# iris.coord_categorisation.add_year(cube, 'time', name='year')
# cube_a = cube.extract(iris.Constraint(depth = 0))
# cube_b = cube.extract(iris.Constraint(longitude = 360-20))

# cube2 = cube_a.aggregated_by('month', iris.analysis.MEAN)
# cube2_b = cube_b.aggregated_by('month', iris.analysis.MEAN)

# months = cube2.coord('month').points

# cube3 = cube_a.data.copy()

# plt_cube = cube_a.copy()
# for i in range(cube_a.shape[0]):
#         month_tmp = cube_a[i].coord('month').points[0]
#         loc = np.where(months == month_tmp)[0][0]
#         cube3[i,:,:] -= cube2[loc].data

# plt_cube.data = cube3

# cube3_b = cube_b.data.copy()
# plt_cube_b = cube_b.copy()
# for i in range(cube_b.shape[0]):
#         month_tmp = cube_b[i].coord('month').points[0]
#         loc = np.where(months == month_tmp)[0][0]
#         cube3_b[i,:,:] -= cube2_b[loc].data

# plt_cube_b.data = cube3_b

# grid_areas = iris.analysis.cartography.area_weights(cube)

# lats = plt_cube.coord('latitude').points
# lons = plt_cube.coord('longitude').points
# lons, lats = np.meshgrid(lons, lats)

# print 'generate filenames'
# import itertools
# letters = ['a','b','c','d','e']
# x = list(itertools.product(letters, repeat = np.size(letters)))


def plot_surface(lons,lats,plt_cube,x,i):
    print i
    plt.figure()
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=-30.0))
    #ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    min_val = -4
    max_val = 4
    data = time_slice.data
    data[ma.where(data >= max_val)] = max_val
    data[ma.where(data <= min_val)] = min_val
    CS = ax.contourf(lons,lats,data,np.linspace(min_val,max_val,31),transform = ccrs.PlateCarree())
    ax.plot([360-20,360-20],[-90,90],'k',linewidth=3,transform = ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)
    ax.coastlines()
    plt.title(np.str(cube[i].coord('month').points[0])+' '+np.str(cube[i].coord('year').points[0]))
    cbar = plt.colorbar(CS,shrink = 0.7)
    cbar.ax.set_xlabel('SST $^o$C')
    #plt.show()
    plt.savefig('/home/ph290/Documents/figures/anim4/sst'+(''.join(x[i]))+'.png',dpi = 200)
    plt.show(block = False)
    plt.close('all')


for i,time_slice in enumerate(plt_cube.slices(['latitude','longitude'])):
    test = glob.glob('/home/ph290/Documents/figures/anim4/sst'+(''.join(x[i]))+'.png')
    if np.size(test) == 0:
	    plot_surface(lons,lats,plt_cube,x,i)


def plot_slice(plt_cube_b,x,i):
	min_val_b = -3
	max_val_b = 7
	data_b = time_slice.data
	data_b[ma.where(data_b >= max_val_b)] = max_val_b
	data_b[ma.where(data_b <= min_val_b)] = min_val_b
	time_slice.data = data_b
	plt.figure()
	qplt.contourf(time_slice,np.linspace(min_val_b,max_val_b,101))
# 	plt.show()
	plt.savefig('/home/ph290/Documents/figures/anim5/temp_slice_'+(''.join(x[i]))+'.png',dpi = 200)
	plt.show(block = False)
	plt.close('all')

for i,time_slice in enumerate(plt_cube_b.slices(['latitude','depth'])):
	test = glob.glob('/home/ph290/Documents/figures/anim5/temp_slice_'+(''.join(x[i]))+'.png')
	if np.size(test) == 0:
		plot_slice(plt_cube_b,x,i)


#this is a completely separte script - move elsewhere...
# import iris
# import iris.quickplot as qplt
# import iris.analysis.cartography
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import numpy as np
# import cartopy
# import numpy.ma as ma

# file = '/home/ph290/Documents/teaching/Anthropogenic_CO2.nc'
# cube = iris.load_cube(file)

# cube.coord('latitude').guess_bounds()
# cube.coord('longitude').guess_bounds()
# cube.coord('depth').guess_bounds()
# grid_areas = iris.analysis.cartography.area_weights(cube)

# cubeb = cube.copy()

# levels = cube.coord('depth').bounds
# for i in np.arange(np.size(levels[:,0])):
# 	cubeb[i].data = np.multiply(cubeb[i].data,levels[i][1]-levels[i][0])

# cube2 = cubeb.collapsed('depth',iris.analysis.SUM)
# #need to multiply each level by it's thickness

# lats = cube2.coord('latitude').points
# lons = cube2.coord('longitude').points
# lons, lats = np.meshgrid(lons, lats)

# plt.figure()
# ax = plt.axes(projection=ccrs.Mollweide(central_longitude=-30.0))
# #ax.set_global()
# min_val = 00.0
# max_val = 1000.0
# data = np.fliplr(np.rot90(cube2.data,3))
# data[ma.where(data >= max_val)] = max_val
# data[ma.where(data <= min_val)] = min_val
# #ax = plt.contourf(lons,lats,data,np.linspace(min_val,max_val,31))

# CS = ax.contourf(lons,lats,data,np.linspace(min_val,max_val,31),transform = ccrs.PlateCarree())
# ax.add_feature(cartopy.feature.LAND)
# ax.add_feature(cartopy.feature.COASTLINE)
# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
# ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
# ax.add_feature(cartopy.feature.RIVERS)
# ax.coastlines()
# cbar = plt.colorbar(CS,shrink = 0.7)
# #cbar.ax.set_xlabel('Anthropogenic Carbon Inventory')
# #plt.show()
# plt.savefig('/home/ph290/Documents/figures/anthro_carb.png',dpi = 500,transparent = True)











