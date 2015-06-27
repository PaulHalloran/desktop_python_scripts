#can you add here the name of a directory where a bunch of text files can go to then be zipped up and sent over to me?
#and use the form below, rather than tilders...
output_directory = ('/home/h04/hadme/..../')

import iris
import numpy as np
import matplotlib.pyplot as plt
import glob
import iris.analysis.cartography
import iris.coord_categorisation
import iris.analysis
import time

def my_callback(cube, field, filename):
        cube.remove_coord('forecast_reference_time')
        cube.remove_coord('forecast_period')
        #the cubes were not merging properly before, because the time coordinate appeard to have teo different names... I think this may work

directory = '/project/qump2/esppe/SRES/'

runs = glob.glob(directory+'*')


stash_code = 'm01s00i001'
lisbon = np.array([39.0,360.0-9.0])
reykjavik = np.array([64.0,360.0-22.0])

print 'working so far...'

for i,run in enumerate(runs):
    print i +' out of about 150'
    run_name = run.split('/')[5]
    run_names.append(run_name)
    cube = iris.load_cube(run+'monthly/*.pp',iris.AttributeConstraint(STASH=stash_code),callback=my_callback)
    lat = cube.coord('latitude')
    lon = cube.coord('longitude')
    lat_coord1 = lat.nearest_neighbour_index(lisbon[0])
    lon_coord1 = lon.nearest_neighbour_index(lisbon[1])
    lat_coord2 = lat.nearest_neighbour_index(reykjavik[0])
    lon_coord2 = lon.nearest_neighbour_index(reykjavik[1])   
    lisbon_mslp = cube.data[:,lat_coord1,lon_coord1].data
    reykjavik_mslp = cube.data[:,lat_coord2,lon_coord2].data
    coord = cube.coord('time')
    year = np.array([coord.units.num2date(value).year for value in coord.points])
    month = np.array([coord.units.num2date(value).month for value in coord.points])
    date = year+month/12.0
    np.savetxt(output_directory + run_name + '.txt',np.vstack((date,lisbon_mslp,reykjavik_mslp)).T,delimiter=',')

