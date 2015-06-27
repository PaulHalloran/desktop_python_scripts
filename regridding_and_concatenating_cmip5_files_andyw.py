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
input_directory = '/data/temp/ph290/andy_w_analysis/'
#Directory where you want to put the processed data. Make sure you have the correct file permissions to write here (e.g. test hat you can make a text file there). Also make sure that you have enough space to save the files (the saved files will probably be of a similar size to what they were before processing).
output_directory = '/data/temp/ph290/andy_w_analysis/processed/'
#comma separated list of the CMIP5 variables that you want to process (e.g. 'tos','fgco2' etc.)
variables = np.array(['tos','sos','intpp','spco2','mlotst','thetao'])
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.
time_period = 'Omon'

'''
Main bit of code follows...
'''

print '****************************************'
print '** this can take a long time (days)   **'
print '** grab a cuppa, but keep an eye on   **'
print '** this to make sure it does not fail **'
print '****************************************'

print 'Processing data from: '+ input_directory
#This runs teh function above to come up with a list of models from the filenames
models = model_names(input_directory)

''' 
in input directory
'''

for model in models:
	var_tmp = []
	files = glob.glob('/data/temp/ph290/andy_w_analysis/processed/'+model+'_*.nc')
	for file in files:
		var_tmp.append(file.split('/')[-1].split('_')[1])
	var_tmp = np.unique(var_tmp)
	if ('spco2' in var_tmp) & ('tos' in var_tmp) & ('sos' in var_tmp):
		if (('mlotst' in var_tmp) or ('mld' in var_tmp)):
			print model+' has all variables'
			files2 = [ x for x in files if not 'thetao' in x ]
			files2 = [ x for x in files2 if not 'intpp' in x ]
			tmp = ' '.join(files2)
			subprocess.call('tar -zcvf /data/temp/ph290/andy_w_analysis/processed/'+model+'.tar.gz '+tmp, shell=True)
		else:
			print model+' has all variables except mld'

			

'''
in output directory
'''
for model in models:
	var_tmp = []
	files = glob.glob(output_directory+'/*'+model+'_*.nc')
	for file in files:
		var_tmp.append(file.split('/')[-1].split('_')[1])
	var_tmp = np.unique(var_tmp)
	if ('spco2' in var_tmp) & ('tos' in var_tmp) & ('sos' in var_tmp):
		if (('mlotst' in var_tmp) or ('mld' in var_tmp)):
			print model+' has all variables'
		else:
			print model+' has all variables except mld'	


'''
NOTE there are very few models that have submitted MLD - so now downloading thetao and will calculate MLD for each model. NOT DONE THIS YET - will take a while to download...
'''

temp_file1 = str(uuid.uuid4())+'.nc'
temp_file2 = str(uuid.uuid4())+'.nc'
temp_file3 = str(uuid.uuid4())+'.nc'			
temp_file4 = str(uuid.uuid4())+'.nc'

#Looping through each model we have
for model in models:
	print model
	dir1 = input_directory
	dir2 = output_directory
	files = glob.glob(dir1+'*'+model+'*.nc')
	#Looping through each variable we have
	for i,var in enumerate(variables):
		subprocess.call('rm '+temporary_file_space+temp_file1, shell=True)
		subprocess.call('rm '+temporary_file_space+temp_file2, shell=True)
		subprocess.call('rm '+temporary_file_space+temp_file3, shell=True)
		subprocess.call('rm '+temporary_file_space+temp_file4, shell=True)
		#testing if the output file has alreday been created
		tmp = glob.glob(dir2+model+'_'+variables[i]+'_regridded.nc')
		temp_file1 = str(uuid.uuid4())+'.nc'
		temp_file2 = str(uuid.uuid4())+'.nc'
		temp_file3 = str(uuid.uuid4())+'.nc'			
		temp_file4 = str(uuid.uuid4())+'.nc'
		if np.size(tmp) == 0:
			#reads in the files to process
			print 'reading in: '+var+'_'+time_period+'_'+model
			files_init = glob.glob(dir1+'/'+var+'*'+time_period+'*_'+model+'_*r1i1p1*.nc')
			#remove any zero=length file names
			files = []
			for file in files_init:
				statinfo = os.stat(file)
				test1 = statinfo.st_size
				if test1 > 0: files.append(file)
			sizing = np.size(files)
			#checks that we have some files to work with for this model and variable
			if not sizing == 0:
				test2 = 0
				if sizing > 1:
					#if the data is split across more than one file, it is combined into a single file for ease of processing
					files = ' '.join(files)
					print 'merging files'
					#merge together different files
					test2 = subprocess.call(['cdo mergetime '+files+' '+temporary_file_space+temp_file1], shell=True)
				if sizing == 1:
					print 'no need to merge - only one file present'
					subprocess.call(['cp '+files[0]+' '+temporary_file_space+temp_file1], shell=True)
				if test2 == 0:
					print 'regridding files horizontally'
					#then regrid data onto a 360x180 grid - you coudl change these values if you wanted to work with different resolurtoin data (lower resolution would make smaller files that would be quicker to work with)
					subprocess.call(['cdo remapbil,r360x180 -selname,'+var+' '+temporary_file_space+temp_file1+' '+temporary_file_space+temp_file2], shell=True)
					print 'regridding files vertically'
					#Moves all of the models on to the smae vertical grid - note, I'm not sure how this will work if some of your models are not using depth levels, but instead (for example) have pressure levels...
					subprocess.call('rm '+temporary_file_space+temp_file1, shell=True)
					#test if we have a depth (or similar) coordinate - i.e. is this 3d?
					cube = iris.load_cube(temporary_file_space+temp_file2)
					coord_names = [np.str(cube.coords()[j].standard_name) for j in range(np.size(cube.coords()))]
					test = (('depth' in coord_names) or ('ocean sigma coordinate' in coord_names) or ('ocean sigma over z coordinate' in coord_names))
					if test:
					#regrid data onto common vertical levels
						subprocess.call(['cdo intlevel,400,200,150,100,80,60,40,30,20,10,5,0 '+temporary_file_space+temp_file2+' '+dir2+model+'_'+variables[i]+'_regridded.nc'], shell=True)
						subprocess.call('rm '+temporary_file_space+temp_file2, shell=True)
					else:
						subprocess.call('mv '+temporary_file_space+temp_file2+' '+dir2+model+'_'+variables[i]+'_regridded.nc', shell=True)
					#read data in using the iris module then anually mean data if it is monthly or daily - again you culd change this if you wanted to work with monthly data
					#By using 'try... except' we are allowing for teh fact that the file might be unreadable for some reason, and are just skipping those models. If skilling it will print the model name and say 'failed' - keep an eye out of this if the files you expect are not produced.
					# try:
					# 	cube = iris.load_cube(temporary_file_space+temp_file3)
					# 	#iris.coord_categorisation.add_year(cube, 'time', name='year2')
					# 	#cube_ann_meaned = cube.aggregated_by('year2', iris.analysis.MEAN)
					# 	print 'writing final file'
					# 	#saves the filly processed file out to the directory you have specified. The resulting file will be a lot easier to work with than the original file
					# 	iris.fileformats.netcdf.save(cube, dir2+model+'_'+variables[i]+'_regridded.nc','NETCDF3_CLASSIC')
					# 	subprocess.call('rm '+temporary_file_space+temp_file4, shell=True)
					# except:
					# 	print model+' failed'
				else:
					print model+' data in file type unsupported by cdo'




