from iris.coords import DimCoord
import iris.plot as iplt
import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy.ma as ma
import running_mean
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d
import running_mean


def index_of_array_items_in_another(x,y):
	index = np.argsort(x)
	sorted_x = x[index]
	sorted_index = np.searchsorted(sorted_x, y)
	yindex = np.take(index, sorted_index, mode="clip")
	mask = x[yindex] != y
	return np.ma.array(yindex, mask=mask)



def digital(cube):
    cube_out = cube.copy()
    cube_out[np.where(cube >= 0.0)] = 1.0
    cube_out[np.where(cube < 0.0)] = -1.0
    return cube_out

#this is a simple function that we call later to look at the file names and extarct from them a unique list of models to process
#note that the model name is in the filename when downlaode ddirectly from the CMIP5 archive
def model_names(directory,variable,experiment):
	files = glob.glob(directory+'/*'+variable+'_'+experiment+'*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models


input_directory = '/media/usb_external1/cmip5/pr_regridded/'

variables = np.array(['pr'])
experiments = ['piControl','past1000','sstClim','sstClimAerosol']

'''
#Main bit of code follows...
'''
	
	
'''
#models' pr sstClim
'''

variable = variables[0]

models1 = model_names(input_directory,variable,experiments[2])
models2 = model_names(input_directory,variable,experiments[3])
models = np.intersect1d(models1,models2)

models = list(models)

#remove models without 1st indirect effect

# models.remove('FGOALS-s2')
# models.remove('bcc-csm1-1')

#remove models without 2nd indirect effect

models.remove('CSIRO-Mk3-6-0')
models.remove('FGOALS-s2')
models.remove('GFDL-CM3')
models.remove('IPSL-CM5A-LR')
models.remove('MRI-CGCM3')
models.remove('bcc-csm1-1')

cube1 = iris.load_cube(input_directory+models[0]+'*'+variable+'*'+experiments[2]+'_*.nc')

models2_sw= []
cubes1 = []
cubes2 = []
cubes_diff = []
high_low = np.empty([np.size(models),np.shape(cube1[0].data)[0],np.shape(cube1[0].data)[1]])
digital_high_low = high_low.copy()

for i,model in enumerate(models):
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory+model+'*'+variable+'_'+experiments[2]+'_*.nc')
	except:
		cube = iris.load(input_directory+model+'*'+variable+'_'+experiments[2]+'_*.nc')
		cube = cube[0]
	cubes1.append(cube.collapsed('time',iris.analysis.MEAN))
	try:
		cubeb = iris.load_cube(input_directory+model+'*'+variable+'_'+experiments[3]+'_*.nc')
	except:
		cubeb = iris.load(input_directory+model+'*'+variable+'_'+experiments[3]+'_*.nc')
		cubeb = cube[0]
	cubes2.append(cubeb.collapsed('time',iris.analysis.MEAN))
	models2_sw.append(model)
	cubes_diff.append(cubes2[i]-cubes1[i])
	high_low[i,:,:] = cubes_diff[i].data
	digital_high_low[i,:,:] = digital(cubes_diff[i].data)
	

high_low_mean_cube = cube1[0].copy()
digital_high_low_mean_cube = cube1[0].copy()
high_low_mean_cube.data = np.mean(high_low,axis = 0)
digital_high_low_mean_cube.data = digital(np.mean(high_low,axis = 0))


#processing digital fields. Count number of models which are +ve where the mean is +ve and -ve where the mean is -ve, then turn that into stippling
digital_data_tmp1 = digital_high_low.copy()
digital_data_tmp2 = digital_high_low.copy()
digital_data_tmp1[np.where(digital_high_low == 1.0)] = 0
digital_low = np.sum(digital_data_tmp1,axis = 0)
digital_data_tmp2[np.where(digital_high_low == -1.0)] = 0
digital_high = np.sum(digital_data_tmp2,axis = 0)
digital_low_cube = cubes1[0].copy()
digital_low_cube.data = digital_low
digital_high_cube = cubes1[0].copy()
digital_high_cube.data = digital_high
digital_low_cube.data[np.where(digital_high_low_mean_cube.data == 1.0)] = 0
digital_high_cube.data[np.where(digital_high_low_mean_cube.data == -1.0)] = 0

digital_cube_final = digital_high_low_mean_cube.copy()
tmp = np.zeros([digital_high_low_mean_cube.shape[0],digital_high_low_mean_cube.shape[1]])
tmp[:] = np.nan
digital_cube_final.data = tmp

#!!!!!!!!!!!#
no_agree = 2
#!!!!!!!!!!!#
digital_cube_final.data[np.where(digital_low_cube.data <= -1.0*no_agree)] = 1
digital_cube_final.data[np.where(digital_high_cube.data >= no_agree)] = 1


#dummy cube
latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
                    units='degrees')
longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
                     units='degrees')
dummy_cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])

digital_cube_final = iris.analysis.interpolate.regrid(digital_cube_final,dummy_cube,mode = 'nearest')

north = 85
south = -20
east = 40
west = -130

tmp1 = high_low_mean_cube.intersection(longitude=(west, east))
tmp1 = tmp1.intersection(latitude=(south, north))

tmp2 = digital_cube_final.intersection(longitude=(west, east))
tmp2 = tmp2.intersection(latitude=(south, north))

plt.close('all')
plt.figure()
min_val = -0.0000025
max_val = +0.0000025
tmp = tmp1.copy()
tmp.data[np.where(tmp.data < min_val)] = min_val
tmp.data[np.where(tmp.data > max_val)] = max_val
qplt.contourf(tmp,np.linspace(min_val,max_val,51))
points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
plt.gca().coastlines()
plt.title('precip: sstClimAero - sstClim amo >= '+np.str(no_agree)+'of3')
plt.savefig('/home/ph290/Documents/figures/pr_sstclim_sstclimaero_2st_indirect.pdf')
plt.show()







