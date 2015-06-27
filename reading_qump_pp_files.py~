import iris
import matplotlib.pyplot as plt


# run with: execfile('/net/home/h04/hador/python_scripts/reading_pp_files.py')

#various ways to read in data from a single stash into iris:

#way 1:
def my_callback(cube, field, filename):
    if field.lbuser[3] != 30249:
        raise iris.exceptions.IgnoreCubeException()

filename='/project/obgc/qump/aldpp/*.pp'
temp_cubes1 = iris.load(filename, callback=my_callback)

#way 2:
temp_cubes2 = iris.load(filename, iris.AttributeConstraint(STASH='m02s30i30249'))


'''
plotting that data simply:
'''

mean_data = temp_cubes1[0].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
#meaning the data

ticks = (mean_data.coord("time").points/24.0/360.0)+1970.0
#working out time axis (note print mean_data.coord("time") tells you the time format is 'hours since 1970-01-01 00:00:00', calendar='360_day')

values = mean_data.data
plt.plot(ticks, values)
plt.show()

