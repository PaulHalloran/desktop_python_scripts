'''
This script processes ocean data (and I think it should work with at least surface level atmospheric data) from teh CMIP5 archive to put it all on the same grid (same horizontal and vertical resolution (nominally 360x180 i.e. 1 degree horizontal, and every 10m in the shallow ocean, and every 500m in the deep ocean - see later if you want to adjust this)). It also convertsaverages monthly data to annual data - again, you could change this.
'''

#First we need to import the modules required to do the analysis
import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os

'''
******************************
'''
#time.sleep(60.0*60.0*24.0*3.0)
'''
******************************
'''

#this is a simple function that we call later to look at the file names and extarct from them a unique list of models to process
#note that the model name is in the filename when downlaode ddirectly from the CMIP5 archive
def model_names(directory):
	files = glob.glob(directory+'/*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[2])
			models = np.unique(models_tmp)
	return models

'''
Defining directory locations, variable locatoins etc.
You may well need to edit the text within the quotation marks to adapt he script to work with your data
'''

#the lcoation of some temporart disk space that can be used for the processing. You'll want to point to an area with plenty of free space (Many Gbs)
temporary_file_space = '/data/temp/ph290/tmp/'
#Directory containing the datasets you want to process onto a simply grid
input_directory = '/data/temp/ph290/reynolds_data/past1000/'
#Directory where you want to put the processed data. Make sure you have the correct file permissions to write here (e.g. test hat you can make a text file there). Also make sure that you have enough space to save the files (the saved files will probably be of a similar size to what they were before processing).
output_directory = '/media/usb_external1/cmip5/reynolds_data/past1000/'
#comma separated list of the CMIP5 experiments that you want to process (e.g. 'historical','rcp85' etc.). Names must be as they are referencedwritted in the filename as downloaded from CMIP5
#experiments = ['piControl']
experiments = ['past1000']
#comma separated list of the CMIP5 variables that you want to process (e.g. 'tos','fgco2' etc.)
variables = np.array(['vo','tos','sos','msftbarot','sic'])
variables = np.array(['pr','psl','sic'])
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.
time_period = 'mon'

'''
Main bit of code follows...
'''

#This is just a bit of tidying up in case to make sure that the temporary file_space we will use for processing the data is free
subprocess.call('rm '+temporary_file_space+'temp2.nc', shell=True)
subprocess.call('rm '+temporary_file_space+'temp.nc', shell=True)

print '****************************************'
print '** this can take a long time (days)   **'
print '** grab a cuppa, but keep an eye on   **'
print '** this to make sure it does not fail **'
print '****************************************'

print 'Processing data from: '+ input_directory
#This runs teh function above to come up with a list of models from the filenames
models = model_names(input_directory)

#bad_models = ['CMCC-CESM','CMCC-CSM','CMCC-CM','CMCC-CMS','CNRM-CM5','CNRM-CM5-2','IPSL-CM5A-MR','GFDL-ESM2M']
bad_models = ['no_models']

#Looping through each model we have
for model in models:
	print model
	#looping through each experiment we have
	if model in bad_models:
		print 'skipping '+model+' as requested'
	if not model in bad_models:
		for experiment in experiments:
			print experiment
			dir1 = input_directory
			dir2 = output_directory
			files = glob.glob(dir1+'*'+model+'*.nc')
			#Looping through each variable we have
			for i,var in enumerate(variables):
				subprocess.call('rm '+temporary_file_space+'temp.nc', shell=True)
				#testing if the output file has alreday been created
				tmp = glob.glob(dir2+model+'_'+variables[i]+'_'+experiment+'_regridded.nc')
				if np.size(tmp) == 0:
					#reads in the files to process
					print 'reading in: '+var+'_'+time_period+'_'+model
					files = glob.glob(dir1+'/'+var+'*'+time_period+'*_'+model+'_*'+experiment+'_*r1i1p1*.nc')
					sizing = np.size(files)
					#checks that we have some files to work with for this model, experiment and variable
					if not sizing == 0:
						if sizing > 1:
							#if the data is split across more than one file, it is combined into a single file for ease of processing
							files = ' '.join(files)
							print 'merging files'
							#merge together different files from the same experiment
							subprocess.call(['cdo mergetime '+files+' '+temporary_file_space+'temp.nc'], shell=True)
						if sizing == 1:
							print 'no need to merge - only one file present'
							subprocess.call(['cp '+files[0]+' '+temporary_file_space+'temp.nc'], shell=True)
						statinfo = os.stat(temporary_file_space+'temp.nc')
						if statinfo.st_size >= 1:
							print 'regridding files horizontally'
							#then regrid data onto a 360x180 grid - you coudl change these values if you wanted to work with different resolurtoin data (lower resolution would make smaller files that would be quicker to work with)
							subprocess.call(['cdo remapbil,r360x180 -selname,'+var+' '+temporary_file_space+'temp.nc '+temporary_file_space+'temp2.nc'], shell=True)
							print 'regridding files vertically'
							#Moves all of the models on to the smae vertical grid - note, I'm not sure how this will work if some of your models are not using depth levels, but instead (for example) have pressure levels...
							subprocess.call('rm '+temporary_file_space+'temp.nc', shell=True)
							#test if we have a depth (or similar) coordinate - i.e. is this 3d?
							cube = iris.load_cube(temporary_file_space+'temp2.nc')
							coord_names = [np.str(cube.coords()[j].standard_name) for j in range(np.size(cube.coords()))]
							test = (('depth' in coord_names) or ('ocean sigma coordinate' in coord_names) or ('ocean sigma over z coordinate' in coord_names))
							if test:
							#regrid data onto common vertical levels
								subprocess.call(['cdo intlevel,6000,5500,5000,4500,4000,3500,3000,2500,2000,1500,1000,900,800,700,600,500,400,300,200,100,90,80,70,60,50,40,30,20,10,5,0 '+temporary_file_space+'temp2.nc '+temporary_file_space+'temp3.nc'], shell=True)
								subprocess.call('rm '+temporary_file_space+'temp2.nc', shell=True)
							else:
								subprocess.call('mv '+temporary_file_space+'temp2.nc '+temporary_file_space+'temp3.nc', shell=True)
							print 'time-meaning data (to annual means if required)'	
							subprocess.call('cdo yearmean '+temporary_file_space+'temp3.nc '+temporary_file_space+'temp4.nc', shell=True)
							subprocess.call('rm '+temporary_file_space+'temp3.nc', shell=True)
							#read data in using the iris module then anually mean data if it is monthly or daily - again you culd change this if you wanted to work with monthly data
							#By using 'try... except' we are allowing for teh fact that the file might be unreadable for some reason, and are just skipping those models. If skilling it will print the model name and say 'failed' - keep an eye out of this if the files you expect are not produced.
							try:
								cube = iris.load_cube(temporary_file_space+'temp4.nc')
							#iris.coord_categorisation.add_year(cube, 'time', name='year2')
							#cube_ann_meaned = cube.aggregated_by('year2', iris.analysis.MEAN)
								print 'writing final file'
							#saves the fully processed file out to the directory you have specified. The resulting file will be a lot easier to work with than the original file
								iris.fileformats.netcdf.save(cube, dir2+model+'_'+variables[i]+'_'+experiment+'_regridded.nc','NETCDF3_CLASSIC')
								subprocess.call('rm '+temporary_file_space+'temp4.nc', shell=True)
							except:
								print model+' failed'


