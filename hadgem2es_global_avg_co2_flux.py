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

directory = '/data/data1/ph290/hadgem2es_co2/n_atlantic_co2/'
output_directory = ('/home/ph290/data1/hadgem2es_co2/global_avg/')

runs = glob.glob(directory+'//?????')

run_names = []
run_global_means = []
run_date = []

for i,run in enumerate(runs):
    print i
    run_name = run.split('/')[7]
    run_names.append(run_name)
    cube = iris.load_cube(run+'/*.pp',iris.AttributeConstraint(STASH='m02s30i249'),callback=my_callback)
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()
    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)
    time_mean = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
    run_global_means.append(time_mean.data)
    coord = cube.coord('time')
    year = np.array([coord.units.num2date(value).year for value in coord.points])
    run_date.append(year)
    np.savetxt(output_directory + run_name + '.txt',np.vstack((year,time_mean.data)).T,delimiter=',')

fig = plt.figure()
for i,data in enumerate(run_global_means):
    plt.plot(run_date[i],i,data,'k')
    plt.xlabel('year')
    plt.ylabel('air-sea CO$_2$ flux')

plt.show()


