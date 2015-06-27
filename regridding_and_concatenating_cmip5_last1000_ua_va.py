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
import uuid


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
	
def ensemble_names(directory):
	files = glob.glob(directory+'/*.nc')
	ensembles_tmp = []
	for file in files:
			statinfo = os.stat(file)
			if statinfo.st_size >= 1:
					ensembles_tmp.append(file.split('/')[-1].split('_')[4])
					ensemble = np.unique(ensembles_tmp)
	return ensemble


'''
Defining directory locations, variable locatoins etc.
You may well need to edit the text within the quotation marks to adapt he script to work with your data
'''

'''
EDIT THE FOLLOWING TEXT
'''
#the lcoation of some temporart disk space that can be used for the processing. You'll want to point to an area with plenty of free space (Many Gbs)
temporary_file_space = '/data/temp/ph290/tmp2/'
#Directory containing the datasets you want to process onto a simply grid
input_directory = '/data/NAS-ph290/ph290/cmip5/past1000/'
#Directory where you want to put the processed data. Make sure you have the correct file permissions to write here (e.g. test hat you can make a text file there). Also make sure that you have enough space to save the files (the saved files will probably be of a similar size to what they were before processing).
output_directory = '/data/NAS-ph290/ph290/cmip5/last1000/'
#comma separated list of the CMIP5 experiments that you want to process (e.g. 'historical','rcp85' etc.). Names must be as they are referencedwritted in the filename as downloaded from CMIP5
experiments = ['past1000']
#comma separated list of the CMIP5 variables that you want to process (e.g. 'tos','fgco2' etc.)
variables = np.array(['ua','va'])
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.
time_periods = np.array(['Amon'])

'''
Main bit of code follows...
'''


print '****************************************'
print '** this can take a long time (days)   **'
print '** grab a cuppa, but keep an eye on   **'
print '** this to make sure it does not fail **'
print '****************************************'

print 'Processing data from: '+ input_directory
#This runs the function above to come up with a list of models from the filenames
models = model_names(input_directory)
ensembles = ensemble_names(input_directory)

models = np.array(['CCSM4'])

#Looping through each model we have
for model in models:
	print model
        for time_period in time_periods:
		for ensemble in ensembles:
			print model + ' ' + ensemble
			#looping through each experiment we have
			for experiment in experiments:
				print experiment
				dir1 = input_directory
				dir2 = output_directory
				files = glob.glob(dir1+'*'+model+'*'+ensemble+'*.nc')
				#Looping through each variable we have
				for i,var in enumerate(variables):
					#testing if the output file has alreday been created
					tmp = glob.glob(dir2+model+'_'+variables[i]+'_'+experiment+'*'+ensemble+'_regridded'+'*'+'.nc')
					#These lines (and similar later on) just create unique random filenames to be used as temporary filenames during the processing
					temp_file1 = str(uuid.uuid4())+'.nc'
					temp_file2 = str(uuid.uuid4())+'.nc'
					temp_file3 = str(uuid.uuid4())+'.nc'			
					temp_file3b = str(uuid.uuid4())+'.nc'
					if np.size(tmp) == 0:
						#reads in the files to process
						print 'reading in: '+var+'_'+time_period+'_'+model
						files = glob.glob(dir1+'/'+var+'*'+time_period+'*_'+model+'_*'+experiment+'*'+ensemble+'*.nc')
						sizing = np.size(files)
						#checks that we have some files to work with for this model, experiment and variable
						if not sizing == 0:
							if sizing > 1:
								#if the data is split across more than one file, it is combined into a single file for ease of processing
								files = ' '.join(files)
								print 'merging files'
								#merge together different files from the same experiment
								subprocess.call(['cdo -P 6 mergetime '+files+' '+temporary_file_space+temp_file1], shell=True)
							if sizing == 1:
								print 'no need to merge - only one file present'
								subprocess.call(['cp '+files[0]+' '+temporary_file_space+temp_file1], shell=True)
							print 'merging months to years'
							subprocess.call('cdo -P 6 yearmean '+temporary_file_space+temp_file1+' '+temporary_file_space+temp_file2, shell=True)
							subprocess.call('rm '+temporary_file_space+temp_file1, shell=True)
							print 'regridding files horizontally'
							#then regrid data onto a 360x180 grid - you could change these values if you wanted to work with different resolurtoin data (lower resolution would make smaller files that would be quicker to work with)
							subprocess.call(['cdo -P 6 remapbil,r360x180 -selname,'+var+' '+temporary_file_space+temp_file2+' '+dir2+model+'_'+variables[i]+'_'+experiment+'_'+ensemble+'_regridded.nc'], shell=True)
							subprocess.call('rm '+temporary_file_space+temp_file2, shell=True)
						else:
							print 'No variable input files for this model'
					else:
						print 'file already exists'	


