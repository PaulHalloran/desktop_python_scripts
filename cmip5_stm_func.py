'''

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
from datetime import datetime, timedelta
import running_mean
import cartopy.crs as ccrs
import iris.analysis.cartography

#execfile('cmip5_stm_func.py')

'''
#producing an Atlantic mask (mask masked and Atlantic has value of 1, elsewhere zero)
'''

input_file = '/media/usb_external1/cmip5/last1000/CCSM4_vo_past1000_regridded.nc'
cube = iris.load_cube(input_file)

tmp_cube = cube[0,16].copy()
tmp_cube = tmp_cube*0.0

location = -30

for y in np.arange(180):
    flag = 0
    tmp = tmp_cube.data.mask[y,:]
    tmp2 = tmp_cube.data[y,:]
    for x in np.arange(360):
        if tmp[location] == True:
            flag = 1
        if ((tmp[location] == False) & (flag == 0)):
            tmp2[location] = 1
        tmp = np.roll(tmp,+1)
        tmp2 = np.roll(tmp2,+1)
    tmp_cube.data.mask[y,:] = tmp
    tmp_cube.data.data[y,:] = tmp2.data

location = location+1

for y in np.arange(180):
    flag = 0
    tmp = tmp_cube.data.mask[y,:]
    tmp2 = tmp_cube.data[y,:]
    for x in np.arange(360):
        if tmp[location] == True:
            flag = 1
        if ((tmp[location] == False) & (flag == 0)):
            tmp2[location] = 1
        tmp = np.roll(tmp,-1)
        tmp2 = np.roll(tmp2,-1)
    tmp_cube.data.mask[y,:] = tmp
    tmp_cube.data.data[y,:] = tmp2.data

tmp_cube.data.data[150:180,:] = 0.0
tmp_cube.data.data[0:40,:] = 0.0
tmp_cube.data.data[:,20:180] = 0.0
tmp_cube.data.data[:,180:280] = 0.0

loc = np.where(tmp_cube.data.data == 0.0)
tmp_cube.data.mask[loc] = True

mask1 = tmp_cube.data.mask


# plt.close('all')
# qplt.contourf(cube[0,28])
# plt.show(block = False)


'''
#calculating stream function
'''

files = glob.glob('/media/usb_external1/cmip5/last1000/*_vo_*.nc')

models = []
strm_fun_26 = []
strm_fun_45 = []
model_years = []


for file in files:
    model = file.split('/')[5].split('_')[0]
    if model not in ('bcc-csm1-1 FGOALS-gl CCSM4'):
    # bcc has a problem with year numbers, FGOALS-g2 is the fast version of FGOALS and we've already got the normal version, and CCSM4 only seems to have about 300 years worth of data...
		print model
		models.append(model)
		cube = iris.load_cube(file)
		coord = cube.coord('time')
		dt = coord.units.num2date(coord.points)
		years = np.array([coord.units.num2date(value).year for value in coord.points])
		model_years.append(years)

		print 'applying mask'
		for level in np.arange(cube.coord('depth').points.size):
			print 'level: '+str(level)
			for year in np.arange(cube.coord('time').points.size):
				print 'year: '+str(year)
				mask2 = cube.data.mask[year,level,:,:]
				tmp_mask = np.ma.mask_or(mask1, mask2)
				cube.data.mask[year,level,:,:] = tmp_mask

		shape = np.shape(cube)

		tmp = cube[0].collapsed('longitude',iris.analysis.SUM)

		tmp = tmp.data.copy()
		collapsed_data = np.tile(tmp,[shape[0],1,1])

		cube.coord('latitude').guess_bounds()
		cube.coord('longitude').guess_bounds()
		grid_areas = iris.analysis.cartography.area_weights(cube[0])
		grid_areas = np.sqrt(grid_areas)
		#doing this because want to weight by width of grid box not area
		print 'collapsing cube along longitude'
		for i,t_slice in enumerate(cube.slices(['depth', 'latitude','longitude'])):
			print i
			collapsed_data[i] = t_slice.collapsed('longitude',iris.analysis.SUM,weights=grid_areas).data

		depths = cube.coord('depth').points
		thicknesses_tmp = (depths[0:-2]-depths[1:-1])/2.0
		thicknesses = thicknesses_tmp[0:-2]+thicknesses_tmp[1:-1]

		thicknesses = np.array([  500.,  500.,  500.,  500.,  500.,  500.,  500.,  500.,  500.,
						500.,  100.,  100.,  100.,  100.,  100.,  100.,  100.,  100.,
						 100.,   10.0,10.,   10.,   10.,   10.,   10.,   10.,   10.,   10., 5.0, 5.0, 0.0])

		thicknesses = thicknesses*1.0

		print 'calculating stream function'
		data = np.zeros([31,180])
		tmp_strm_fun_45 = []
		tmp_strm_fun_26 = []

		loc = np.where(cube[0].coord('latitude').points >= 45)[0]
		loc2 = np.where(cube[0].coord('latitude').points >= 26)[0]

		for i in np.arange(shape[0]):
			print i
			tmp_data = collapsed_data[i]*np.flipud(np.rot90(np.tile(thicknesses,(180,1))))
			data[:,:] = np.cumsum(np.flipud(tmp_data),axis = 1)
			tmp_data_2 = data[:,loc[0]]
			tmp_strm_fun_45 = np.append(np.max(tmp_data_2/1.0e6),tmp_strm_fun_45)
			tmp_data_3 = data[:,loc2[0]]
			tmp_strm_fun_26 = np.append(np.max(tmp_data_3/1.0e6),tmp_strm_fun_26) 
			#I SEEM TO BE ABOUT A FACTOR OF TWO OUT WHERE HAS THIS COME FROM???


		strm_fun_26.append(tmp_strm_fun_26)
		strm_fun_45.append(tmp_strm_fun_45)

# 'print removing too short data - submitted wrong simulation?
# models.pop(0)
# model_years.pop(0)
# strm_fun_26.pop(0)
# strm_fun_45.pop(0)

'''

############
#Read in models that have submitted strm fun diagnostics
############

my_dir = '/media/usb_external1/cmip5/msftmyz/last1000/'
files1 = np.array(glob.glob(my_dir+'*msftmyz*.nc'))


models1 = []
for file in files1:
	models1.append(file.split('/')[-1].split('_')[0])

models_unique = np.unique(np.array(models1))

cmip5_max_strmfun_26 = []
cmip5_max_strmfun_45 = []
cmip5_year = []

models_unique = models_unique.tolist()
#models_unique.remove('MRI-CGCM3')

for model in models_unique:
	print model
	files = np.array(glob.glob(my_dir+'/*'+model+'_*msftmy*.nc'))
	cube = iris.load_cube(files)[:,0,:,:]
	loc = np.where(cube.coord('grid_latitude').points >= 26)[0]
	lat = cube.coord('grid_latitude').points[loc[0]]
	sub_cube = cube.extract(iris.Constraint(grid_latitude = lat))
	stream_function_tmp = sub_cube.collapsed('depth',iris.analysis.MAX)
	coord = stream_function_tmp.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	tmp = stream_function_tmp.data/1.0e9
	cmip5_max_strmfun_26.append(tmp[np.logical_not(np.isnan(tmp))])
	cmip5_year.append(year_tmp[np.logical_not(np.isnan(tmp))])

for model in models_unique:
	print model
	files = np.array(glob.glob(my_dir+'/*'+model+'_*msftmy*.nc'))
	cube = iris.load_cube(files)[:,0,:,:]
	loc = np.where(cube.coord('grid_latitude').points >= 45)[0]
	lat = cube.coord('grid_latitude').points[loc[0]]
	sub_cube = cube.extract(iris.Constraint(grid_latitude = lat))
	stream_function_tmp = sub_cube.collapsed('depth',iris.analysis.MAX)
	coord = stream_function_tmp.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	tmp = stream_function_tmp.data/1.0e9
	cmip5_max_strmfun_45.append(tmp[np.logical_not(np.isnan(tmp))])

cmip5_max_strmfun_26 = np.array(cmip5_max_strmfun_26)
cmip5_max_strmfun_45 = np.array(cmip5_max_strmfun_45)
cmip5_year = np.array(cmip5_year)


size = 0
years = 0
for item in model_years:
	if np.size(item) > size:
		years =  item
	size = np.max([size,np.size(item)])

output = np.empty([size,np.size(models)*2+1+2])
output[:] = np.NAN
    
output[:,0] = years



for j,dummy in enumerate(models):
	for k,yr in enumerate(model_years[j]):
		loc = np.where(years == yr)
		if np.size(loc[0]) > 0:
			output[loc[0],j+1] = strm_fun_26[j][k]
			output[loc[0],np.size(models)+j+1] = strm_fun_45[j][k]
			
for k,yr in enumerate(cmip5_year[0]):
	loc = np.where(years == yr)
	if np.size(loc[0]) > 0:
		output[loc[0],-2] = cmip5_max_strmfun_26[0][k]
		output[loc[0],-1] = cmip5_max_strmfun_45[0][k]


output = np.ma.masked_invalid(output)

header_txt = 'year'
for model in models:
	header_txt += ','+model+' 26N'

for model in models:
	header_txt += ','+model+' 45N'

fid = open( '/home/ph290/data0/cmip5_data/strmfun/cmip5_strm_function_last1000.txt' , 'w' ) 
fid.write( header_txt+'\n' ) 
np.savetxt(fid, output, delimiter=',')
fid.close() 

mean_srm_fun_26 = np.mean(output[:,1:3],axis = 1)
mean_srm_fun_45 = np.mean(output[:,3:6],axis = 1)

model_names = ['year']
for model in models:
	model_names.append(np.str(model)+' 26N')
	
for model in models:
	model_names.append(np.str(model)+' 45N')
	
model_names.append('mpi 26')
model_names.append('mpi 45')
	
import running_mean
plt.close('all')
plt.figure(figsize = (15,5),dpi = 75)
for i in [1,2,3,4,7]:
	plt.plot(output[:,0],running_mean.running_mean(output[:,i]/np.mean(output[:,i]),20),linewidth = 3,alpha = 0.5,label = model_names[i])
	
plt.plot(output[:,0],running_mean.running_mean(mean_srm_fun_26/np.mean(mean_srm_fun_26),20),'k',linewidth = 3,alpha = 0.5,label = 'mean')

leg = plt.legend()
leg.get_frame().set_alpha(0.5)

plt.show(block = False)



# plt.close('all')
# plt.plot(cmip5_year[0],running_mean.running_mean(cmip5_max_strmfun_26[0],20))
# plt.plot(model_years[-1],running_mean.running_mean(strm_fun_26[2]/4,20))
# plt.show(block = False)
