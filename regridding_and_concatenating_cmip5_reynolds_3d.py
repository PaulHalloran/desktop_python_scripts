'''
This script processes data from the CMIP5 archive to put it all on the same grid (same horizontal resolution (nominally 360x180 i.e. 1 degree horizontal - see later if you want to adjust this)). It also converts file formats and averages monthly data to annual data - again, you could change this if you want monthly data and have such data to start with. I have a version of this script that can do vertical regridding, so let me know if you need this.
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
temporary_file_space = '/home/ph290/data1/temp/'
#Directory containing the datasets you want to process onto a simply grid
input_directory = '/data/NAS-ph290/ph290/data_for_reynolds_v2/'
#Directory where you want to put the processed data. Make sure you have the correct file permissions to write here (e.g. test hat you can make a text file there). Also make sure that you have enough space to save the files (the saved files will probably be of a similar size to what they were before processing).
output_directory = '/data/NAS-ph290/ph290/data_for_reynolds_v2/regridded/'
#comma separated list of the CMIP5 experiments that you want to process (e.g. 'historical','rcp85' etc.). Names must be as they are referencedwritted in the filename as downloaded from CMIP5
experiments = ['piControl']
#comma separated list of the CMIP5 variables that you want to process (e.g. 'tos','fgco2' etc.)
variables = np.array(['thetao'])
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.
time_periods = ['Omon']

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

#These lines (and similar later on) just create unique random filenames to be used as temporary filenames during the processing
temp_file1 = str(uuid.uuid4())+'.nc'
temp_file2 = str(uuid.uuid4())+'.nc'
temp_file3 = str(uuid.uuid4())+'.nc'			
temp_file4 = str(uuid.uuid4())+'.nc'

#Looping through each model we have
for model in models:
	print model
	for time_period in time_periods:
		print time_periods
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
					subprocess.call('rm '+temporary_file_space+temp_file1, shell=True)
					subprocess.call('rm '+temporary_file_space+temp_file2, shell=True)
					subprocess.call('rm '+temporary_file_space+temp_file3, shell=True)
					subprocess.call('rm '+temporary_file_space+temp_file4, shell=True)
					#testing if the output file has alreday been created
					tmp = glob.glob(dir2+model+'_'+variables[i]+'_'+experiment+'*'+ensemble+'_regridded'+'*'+'.nc')
					temp_file1 = str(uuid.uuid4())+'.nc'
					temp_file2 = str(uuid.uuid4())+'.nc'
					temp_file3 = str(uuid.uuid4())+'.nc'			
					temp_file4 = str(uuid.uuid4())+'.nc'
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
								subprocess.call(['cdo -P 11 mergetime '+files+' '+temporary_file_space+temp_file1], shell=True)
							if sizing == 1:
								print 'no need to merge - only one file present'
								subprocess.call(['cp '+files[0]+' '+temporary_file_space+temp_file1], shell=True)
							print 'merging months to years'
							subprocess.call('cdo -P 11 yearmean '+temporary_file_space+temp_file1+' '+temporary_file_space+temp_file2, shell=True)
							subprocess.call('rm '+temporary_file_space+temp_file1, shell=True)
							print 'regridding files horizontally'
							#then regrid data onto a 360x180 grid - you could change these values if you wanted to work with different resolurtoin data (lower resolution would make smaller files that would be quicker to work with)
							subprocess.call(['cdo -P 11 remapbil,r360x180 -selname,'+var+' '+temporary_file_space+temp_file2+' '+temporary_file_space+temp_file3], shell=True)
							subprocess.call('rm '+temporary_file_space+temp_file2, shell=True)
							test = False
							proc = subprocess.Popen('ncdump -h '+temporary_file_space+temp_file3,shell=True,stdout=subprocess.PIPE)
							for line in proc.stdout:
								if '"depth"' in line:
									test = True
								if '"ocean sigma coordinate"' in line:
									test = True
								if '"ocean sigma over z coordinate"' in line:
									test = True


							if test:
							#regrid data onto common vertical levels
								subprocess.call(['cdo -P 11 intlevel,6000,5500,5000,4500,4000,3500,3000,2500,2000,1500,1000,900,800,700,600,500,400,300,200,100,90,80,70,60,50,40,30,20,10,5,0 '+temporary_file_space+temp_file3+' '+dir2+model+'_'+variables[i]+'_'+experiment+'_'+ensemble+'_regridded.nc'], shell=True)
							else:
								subprocess.call('mv '+temporary_file_space+temp_file3+' '+dir2+model+'_'+variables[i]+'_'+experiment+'_'+ensemble+'_regridded.nc', shell=True)

							#Moves all of the models on to the smae vertical grid - note, I'm not sure how this will work if some of your models are not using depth levels, but instead (for example) have pressure levels...
							subprocess.call('rm '+temporary_file_space+temp_file3, shell=True)


