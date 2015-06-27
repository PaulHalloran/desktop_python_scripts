import iris
import matplotlib.pyplot as plt
import iris.analysis
import iris.quickplot as qplt

filename='Downloads/dissolved_oxygen_annual_1deg.nc'
cubes=iris.load(filename)
print cubes
cube=iris.load_cube(filename,'Objectively Analyzed Climatology')
cube=cube.collapsed('time',iris.analysis.MEAN)
level=cube.extract(iris.Constraint(depth=0))
print cube.coord('longitude').points

meridional=cube.extract(iris.Constraint(longitude=180.5))


meridional.coord('depth').points=meridional.coord('depth').points*(-1.0)
qplt.contourf(meridional,100,coords=['latitude','depth'])
plt.show()
