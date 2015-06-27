import iris
import iris.analysis
import iris.coord_categorisation


def monthly_to_yearly(cube):
	iris.coord_categorisation.add_year(cube, 'time', name='year2')
	cube_tmp = cube.aggregated_by('year2', iris.analysis.MEAN)
	return cube_tmp


