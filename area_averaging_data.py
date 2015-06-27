import iris
import iris.analysis.cartography
import iris.analysis


#Using a simple grid:

filename = iris.sample_data_path('ostia_monthly.nc')
cube = iris.load_cube(filename, 'surface_temperature')
cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(cube)
w_cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

#using crazy CMIP5 grid, along with its area cell-o file

file_name='/project/champ/data/cmip5/output1/MPI-M/MPI-ESM-MR/historical/yr/ocnBgchem/Oyr/r1i1p1/v20120503/talk/talk_Oyr_MPI-ESM-MR_historical_r1i1p1_1860-1869.nc'
areacello_file='/project/champ/data/cmip5/output1/MPI-M/MPI-ESM-MR/rcp85/fx/ocean/fx/r0i0p0/v20120503/areacello/areacello_fx_MPI-ESM-MR_rcp85_r0i0p0.nc'

cube=iris.load_cube(file_name)
areacello_cube=iris.load_cube(areacello_file)
#these are cell areas - not aure what 'weights' is in iris-talk
surface_slice = cube.extract(iris.Constraint(depth=0))
#extract just the surface level

weights=areacello_cube.data

weights2=np.tile(weights,(10,1,1))
#replicates array, so same number of elements as cube.data (test with cube.data.shape)

area_avged_cube = surface_slice.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=weights2)

print area_avged_cube.data
#gives [2.42607406734 2.42578128696 2.42610900833 2.42612988061 2.42578149796
# 2.42609495335 2.42626334571 2.42651670744 2.4262844869 2.42637880637] - this is correct according to IDL!

