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
#variables = np.array(['wfo','sfdsi','vsf','friver','pr','prsn','evs','fsitherm','mlotst','sos','tauu','tauv','vo','uo','so'])
#variables = np.array(['thetao','so','vo','uo'])
variables = np.array(['uo','vo'])

#later add va and ua``
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.
time_periods = np.array(['Omon'])

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

#models = ['HadCM3','MRI-CGCM3', 'MPI-ESM-P', 'GISS-E2-R','CSIRO-Mk3L-1-2', 'MIROC-ESM', 'CCSM4']
models = ['MIROC-ESM']


files = glob.glob(input_directory+'uo_*MRI*')
for i,file in enumerate(files):
	print i
	subprocess.call('cdo sellevidx,1 '+file+' '+file+'_tmp', shell=True)


files = glob.glob(input_directory+'uo_*MRI*_tmp')
files = ' '.join(files)
subprocess.call(['cdo -P 3 mergetime '+files+' /data/temp/ph290/tmp2/tmp_delete.nc'], shell=True)
run this next:
subprocess.call(['cdo -P 3 remapbil,r360x180 /data/temp/ph290/tmp2/tmp_delete.nc /data/NAS-ph290/ph290/cmip5/last1000/MRI-CGCM3_uo_past1000_r1i1p1_regridded_not_vertically_Omon.nc'], shell=True)




-------------



files = glob.glob(input_directory+'uo_*MIROC-ESM*.nc')
for i,file in enumerate(files):
	print i
	subprocess.call('cdo selstdname,sea_water_x_velocity '+file+' /data/temp/ph290/tmp2/tmp_delete4.nc', shell=True)
	subprocess.call('cdo sellevidx,1 /data/temp/ph290/tmp2/tmp_delete4.nc '+file+'_tmp1', shell=True)



files = glob.glob(input_directory+'uo_*MIROC-ESM*_tmp1')
files = ' '.join(files)
subprocess.call(['cdo -P 3 mergetime '+files+' /data/temp/ph290/tmp2/tmp_delete2.nc'], shell=True)
subprocess.call(['cdo -P 3 yearmean /data/temp/ph290/tmp2/tmp_delete2.nc /data/temp/ph290/tmp2/tmp_delete3.nc'], shell=True)
subprocess.call(['cdo -P 3 remapbil,r360x180 /data/temp/ph290/tmp2/tmp_delete3.nc /data/NAS-ph290/ph290/cmip5/last1000/MIROC-ESM_uo_past1000_r1i1p1_regridded_not_vertically_Omon.nc'], shell=True)



----------------

subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete2.nc', shell=True)
subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete3.nc', shell=True)
subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete4.nc', shell=True)

input_directory = '/data/NAS-ph290/ph290/cmip5/past1000/'

files = glob.glob(input_directory+'vo_*MIROC-ESM*.nc')
for i,file in enumerate(files):
	print i
	subprocess.call('cdo selstdname,sea_water_y_velocity '+file+' /data/temp/ph290/tmp2/tmp_delete4.nc', shell=True)
	subprocess.call('cdo sellevidx,1 /data/temp/ph290/tmp2/tmp_delete4.nc '+file+'_tmp1', shell=True)



files = glob.glob(input_directory+'vo_*MIROC-ESM*_tmp1')
files = ' '.join(files)
subprocess.call(['cdo -P 3 mergetime '+files+' /data/temp/ph290/tmp2/tmp_delete2.nc'], shell=True)
subprocess.call(['cdo -P 3 yearmean /data/temp/ph290/tmp2/tmp_delete2.nc /data/temp/ph290/tmp2/tmp_delete3.nc'], shell=True)
subprocess.call(['cdo -P 3 remapbil,r360x180 /data/temp/ph290/tmp2/tmp_delete3.nc /data/NAS-ph290/ph290/cmip5/last1000/MIROC-ESM_vo_past1000_r1i1p1_regridded_not_vertically_Omon.nc'], shell=True)

--------------------

subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete2b.nc', shell=True)
subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete3b.nc', shell=True)
subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete4b.nc', shell=True)


input_directory = '/data/NAS-ph290/ph290/cmip5/past1000/'

files = glob.glob(input_directory+'uo_*MRI-CGCM3*.nc')
for i,file in enumerate(files):
	print i
	subprocess.call('cdo selstdname,sea_water_x_velocity '+file+' /data/temp/ph290/tmp2/tmp_delete4b.nc', shell=True)
	subprocess.call('cdo sellevidx,1 /data/temp/ph290/tmp2/tmp_delete4b.nc '+file+'_tmp1', shell=True)



files = glob.glob(input_directory+'uo_*MRI-CGCM3*_tmp1')
files = ' '.join(files)
subprocess.call(['cdo -P 3 mergetime '+files+' /data/temp/ph290/tmp2/tmp_delete2b.nc'], shell=True)
subprocess.call(['cdo -P 3 yearmean /data/temp/ph290/tmp2/tmp_delete2b.nc /data/temp/ph290/tmp2/tmp_delete3b.nc'], shell=True)
subprocess.call(['cdo -P 3 remapbil,r360x180 /data/temp/ph290/tmp2/tmp_delete3b.nc /data/NAS-ph290/ph290/cmip5/last1000/MRI-CGCM3_uo_past1000_r1i1p1_regridded_not_vertically_Omon.nc'], shell=True)

subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete2b.nc', shell=True)
subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete3b.nc', shell=True)
subprocess.call('rm /data/temp/ph290/tmp2/tmp_delete4b.nc', shell=True)



