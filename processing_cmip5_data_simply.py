import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess

subprocess.call('rm /home/data/tmp/temp2.nc', shell=True)
subprocess.call('rm /home/data/tmp/temp.nc', shell=True)

model = 'hadgem2es'

runs = ['historical','rcp85']
vars = np.array(['intpp','fgco2','thetao','so','uo','vo','evspsbl','pr','intpcalc','zos','dissic','talk','o2','no3','si','po4'])
var_names =np.array(['primary_production','air_sea_co2_flux','potential_temperature','salinity','zonal_velocity','meridional_velocity','evaporation','precipitation','calcification','dissolved_inorganic_carbon','total_alkalinity','oxygen','nitrate','silicate','phosphate'])

for run in runs:
	dir = '/home/data/cmip5/'+model+'/'
	files = glob.glob(dir+'*'+run+'*.nc')
	vars2 = []
	for file in files:
		vars2.append(file.split('/')[-1].split('_')[0])
	vars_unique = np.unique(vars2)
	for i,var in enumerate(vars):
		if np.size(np.where(var == vars_unique)) == 1: 
			files = glob.glob(dir+var+'*'+run+'*.nc')
			if len(files) > 0:
				files = ' '.join(files)
				subprocess.call(['cdo mergetime '+files+' /home/data/tmp/temp.nc'], shell=True)
				subprocess.call(['cdo remapbil,r180x90 -selname,'+var+' /home/data/tmp/temp.nc /home/data/tmp/temp2.nc'], shell=True)
				subprocess.call('rm /home/data/tmp/temp.nc', shell=True)
				cube = iris.load_cube('/home/data/tmp/temp2.nc')
				iris.coord_categorisation.add_year(cube, 'time', name='year2')
				cube_ann_meaned = cube.aggregated_by('year2', iris.analysis.MEAN)
				iris.fileformats.netcdf.save(cube_ann_meaned, dir+'regridded/'+model+'_'+var_names[i]+'_'+run+'_regridded.nc')
				subprocess.call('rm /home/data/tmp/temp2.nc', shell=True)


