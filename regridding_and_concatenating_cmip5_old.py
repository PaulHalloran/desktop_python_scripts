'''
NOTE this is presently set up to output only surface level fields because of memory problems - see sectoin '#test how many levels and if 3D just take top level' to edit that condition
'''

import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os

#print 'waiting'
#time.sleep(60.0*60.0*12.0)
#print 'resuming'

def model_names(directory):
	files = glob.glob(directory+'/*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[2])
			models = np.unique(models_tmp)
	return models

subprocess.call('rm /home/ph290/data0/tmp/temp2.nc', shell=True)
subprocess.call('rm /home/ph290/data0/tmp/temp.nc', shell=True)

directory_tmp = '/media/usb_external1/cmip5/gulf_stream_analysis/'

#runs = ['piControl']
runs = ['rcp85']
variables = np.array(['uo','tas'])
time_period = ['Omon','Amon']

directories =[directory_tmp+runs[0]+'/'+variables[0],directory_tmp+runs[0]+'/'+variables[1]]
#,directory_tmp+runs[1]+'/'+variables[0],directory_tmp+runs[1]+'/'+variables[1]]

models_1 = model_names(directories[0])
models_2 = model_names(directories[1])
#models_3 = model_names(directories[2])
#models_4 = model_names(directories[3])

models = np.intersect1d(models_1,models_2)
#models = np.intersect1d(models,models_3)
#models = np.intersect1d(models,models_4)
#note for this analysis we dont necessrily need them to be in all four - just in each pair in each experiment...

for model in models:
	print model
	for run in runs:
		print run
		dir1 = '/media/usb_external1/cmip5/gulf_stream_analysis/'+run+'/'
		dir2 = '/media/usb_external1/cmip5/gulf_stream_analysis/regridded/'
		files = glob.glob(dir1+'*'+model+'*.nc')
		for i,var in enumerate(variables):
			subprocess.call('rm /home/ph290/data0/tmp/temp.nc', shell=True)
			tmp = glob.glob(dir2+model+'_'+variables[i]+'_'+run+'_regridded.nc')
			if np.size(tmp) == 0:
				files_init = glob.glob(dir1+'/'+var+'*'+time_period+'*_'+model+'_*r1i1p1*.nc')
				#remove any zero=length file names
				files = []
				for file in files_init:
					statinfo = os.stat(file)
					test1 = statinfo.st_size
					if test1 > 0: files.append(file)
				sizing = np.size(files)
				if sizing > 1:
					files = ' '.join(files)
					print 'merging files'
					#merge together different files from the same run
					subprocess.call(['cdo mergetime '+files+' /home/ph290/data0/tmp/temp.nc'], shell=True)
				if sizing == 1:
					print 'no need to merge - only one file present'
					subprocess.call(['cp '+files[0]+' /home/ph290/data0/tmp/temp.nc'], shell=True)
				print 'regridding files'
				#regrid data onto a 360x180 grid
				subprocess.call(['cdo remapbil,r360x180 -selname,'+var+' /home/ph290/data0/tmp/temp.nc /home/ph290/data0/tmp/temp2.nc'], shell=True)
				print 'time-meaning data (to annual means if required)'			
				#read data into iris and anually mean data
				try:
					cube = iris.load_cube('/home/ph290/data0/tmp/temp2.nc')
					#test how many levels and if 3D just take top level
					coord_names = [coord.name() for coord in cube.coords()]
					test1 = np.size(np.where(np.array(coord_names) == 'ocean sigma coordinate'))
					test1b = np.size(np.where(np.array(coord_names) == 'ocean sigma over z coordinate'))
					if test1 == 1:
						cube.coord('ocean sigma coordinate').long_name = 'depth'
					if test1b == 1:
						cube.coord('ocean sigma over z coordinate').long_name = 'depth'
					coord_names = [coord.name() for coord in cube.coords()]
					test2 = np.size(np.where(np.array(coord_names) == 'depth'))
					if test2 == 1:
						cube = cube.extract(iris.Constraint(depth = 0))
						#note - this was 1 for a while - once got threough eveythingf reprocess with 0 for all
					iris.coord_categorisation.add_year(cube, 'time', name='year2')
					cube_ann_meaned = cube.aggregated_by('year2', iris.analysis.MEAN)
					print 'writing final file'
					iris.fileformats.netcdf.save(cube_ann_meaned, dir2+model+'_'+variables[i]+'_'+run+'_regridded.nc','NETCDF3_CLASSIC')
					subprocess.call('rm /home/ph290/data0/tmp/temp2.nc', shell=True)
				except:
					print model+' failed'


