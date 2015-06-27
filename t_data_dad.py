import numpy as np
import iris
import iris.analysis.cartography
import iris.analysis
import matplotlib.pyplot as plt
import iris.quickplot as qplt

file_min='/project/decadal/hadre/HadGHCND_Ex/actual/HadGHCND_TN_acts_50-11_feb12.pp'
cube=iris.load_cube(file_min)
file_max='/project/decadal/hadre/HadGHCND_Ex/actual/HadGHCND_TX_acts_50-11_feb12.pp'
cube2=iris.load_cube(file_max)

temp= cube.extract(iris.Constraint(latitude=52.5, longitude=0))
temp2= cube2.extract(iris.Constraint(latitude=52.5, longitude=0))

cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()
#print cube.coord('latitude').bounds
#print cube.coord('longitude').bounds

#bound are:
#51.25,  53.75
#-1.875    1.875

coord = cube.coord('time')
dt = coord.units.num2date(coord.points)
yr=[]
mn=[]
dy=[]
for i in range(mean.data.size):
    yr.append(dt[i].year)
    mn.append(dt[i].month)
    dy.append(dt[i].day)


np.savetxt('/data/local/hador/t_minb.txt', np.vstack((yr,mn,dy,temp.data)).T, delimiter=',')
np.savetxt('/data/local/hador/t_maxb.txt', np.vstack((yr,mn,dy,temp2.data)).T, delimiter=',')
