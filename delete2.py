lon_west2 = -30+360
lon_east2 = -0+360
lat_south2 = 65
lat_north2 = 80

on2 = iris.Constraint(longitude=lambda v: lon_west2 <= v <= lon_east2,latitude=lambda v: lat_south2 <= v <= lat_north2)

cube = iris.load_cube(input_directory+models[0]+'*'+variables[0]+'*.nc')
cube_region2 = cube.extract(region2)

qplt.contourf(cube_region2[0],30)
plt.show(block = False)


