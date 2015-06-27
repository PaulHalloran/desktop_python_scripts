cube = iris.load_cube(directory+'HadCM3'+'_tos_past1000_r1i1p1*_regridded_not_vertically.nc')


try:
    iris.coord_categorisation.add_year(cube, 'time', name='year2')
except:
    print 'year2 already exists'

    
loc = np.where(cube.coord('year2').points == 1850)
loc2 = cube.coord('time').points[loc[0]]
cube2 = cube.extract(iris.Constraint(time = loc2))
try:
    cube2 = cube2.collapsed('depth',iris.analysis.MEAN)
except:
    print 'only one depth'


coord = cube.coord('time')
dt = coord.units.num2date(coord.points)
years2 = np.array([coord.units.num2date(value).year for value in coord.points])

mltimodel_mean_cube_data_tos = cube.data

for yr in years2:
    print yr
    tmp_array = np.zeros([np.shape(cube2)[0],np.shape(cube2)[1],np.size(models2)])
    tmp_array[:] = np.nan
    for i,model in enumerate(models2):
        cube = iris.load_cube(directory+model+'_tos_past1000_r1i1p1*_regridded_not_vertically.nc')
        try:
            iris.coord_categorisation.add_year(cube, 'time', name='year2')
        except:
            print 'year2 already exists'
        loc = np.where(cube.coord('year2').points == yr)
        if np.size(loc) != 0:
            loc2 = cube.coord('time').points[loc[0]]
            cube2 = cube.extract(iris.Constraint(time = loc2))
            try:
                cube2 = cube2.collapsed('depth',iris.analysis.MEAN)
            except:
                print 'only one depth'
            tmp_array[:,:,i] = cube2.data
    meaned_year = scipy.stats.nanmean(tmp_array,axis = 2)
    mltimodel_mean_cube_data_tos[i,:,:] = meaned_year





