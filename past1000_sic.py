'''
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
import scipy.stats as stats

#this is a simple function that we call later to look at the file names and extarct from them a unique list of models to process
#note that the model name is in the filename when downlaode ddirectly from the CMIP5 archive
def model_names(directory,variable):
	files = glob.glob(directory+'/*'+variable+'*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models


'''
#Main bit of code follows...
'''

lon_west1 = 0
lon_east1 = 360
lat_south1 = 0.0
lat_north1 = 90.0

region1 = iris.Constraint(longitude=lambda v: lon_west1 <= v <= lon_east1,latitude=lambda v: lat_south1 <= v <= lat_north1)


variables = np.array(['sic','pr'])
#specify the temperal averaging period of the data in your files e.g for an ocean file 'Omon' (Ocean monthly) or 'Oyr' (ocean yearly). Atmosphere would be comething like 'Amon'. Note this just prevents probvlems if you accidently did not specify the time frequency when doenloading the data, so avoids trying to puut (e.g.) daily data and monthly data in the same file.


input_directory2 = '/media/usb_external1/cmip5/reynolds_data/past1000/'

modelsb = model_names(input_directory2,variables[0])

cube1 = iris.load_cube(input_directory2+modelsb[0]+'*'+variables[0]+'*.nc')[0]

modelbs2 = []
cubes_n_hem = []
ts_n_hem = []

for model in modelsb:
	print 'processing: '+model
	try:
		cube = iris.load_cube(input_directory2+model+'*'+variables[0]+'*.nc')
	except:
		cube = iris.load(input_directory2+model+'*'+variables[0]+'*.nc')
		cube = cube[0]
	if model == ('MRI-CGCM3'):
		for i in range(cube.shape[0]):
			cube.data.mask[i] = cube1.data.mask
	tmp1 = cube.extract(region1)
	cubes_n_hem.append(tmp1)
	#qplt.contourf(tmp1[0])
	#plt.show()
	ts_n_hem.append(tmp1.collapsed(['latitude','longitude'],iris.analysis.MEAN))

		
mann_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
mann_data = np.genfromtxt(mann_file,skip_header = 4)

mann_file2 = '/home/ph290/data0/misc_data/amoscr.txt'
mann_data2 = np.genfromtxt(mann_file2)

reynolds_file = '/home/ph290/data0/reynolds/ultra_data.csv'
reynolds_data = np.genfromtxt(reynolds_file,skip_header = 1,delimiter = ',')

tmp = np.shape(cubes_n_hem[0].data)
tmp_data = np.ma.empty([np.size(cubes_n_hem),tmp[0],tmp[1],tmp[2]])

for i in np.arange(np.size(cubes_n_hem)):
	tmp_data[i] = cubes_n_hem[i].data[0:1000]


mean_cube = cubes_n_hem[0].copy()
mean_cube.data = np.mean(tmp_data,axis = 0)



coord = ts_n_hem[0].coord('time')
dt = coord.units.num2date(coord.points)
year = np.array([coord.units.num2date(value).year for value in coord.points])


data = np.zeros([1000,np.size(ts_n_hem)])
j=0
for i,ts in enumerate(ts_n_hem):
	print modelsb[i]
	data[:,j] = signal.detrend(ts.data[0:1000])
	j += 1

multimodel_mean = data.mean(axis = 1)
multimodel_max = data.max(axis = 1)
multimodel_min = data.min(axis = 1)


plt.close('all')
fig = plt.subplots(2,1,figsize=(16, 8))

#plt 1
ax1 = plt.subplot(2,1,1)
year2 = year[0]+np.arange(1000)
y = running_mean.running_mean(multimodel_mean-np.mean(multimodel_mean),1)
ax1.plot(year2,y,'b',linewidth = 2, alpha=0.8,label = 'model mean seaice')
x_fill = np.concatenate([year2,np.flipud(year2)])
y_fill = np.concatenate([multimodel_min,np.flipud(multimodel_max)])
ax1.fill_between(x_fill,y_fill,edgecolor = 'none', facecolor='blue', alpha=0.2)

ax2 = ax1.twinx()
loc = np.where((reynolds_data[:,0] <= 1850) & (reynolds_data[:,0] >= 850) )[0]
y_reynolds = reynolds_data[loc[0]:loc[-1],1]
ax2.plot(reynolds_data[loc[0]:loc[-1],0],running_mean.running_mean(signal.detrend(y_reynolds),1),'r',linewidth = 2, alpha=0.5,label = 'd18O')

ax1.set_ylabel('seaice')
ax2.set_ylabel('d18OT')
ax1.set_xlim([900,1900])
ax1.set_ylim([-1.5,1.5])
ax1.plot([900,1900],[0,0],'k')

#plt 2
ax1 = plt.subplot(2,1,2)
lab1, = ax1.plot(year2,y,'b',linewidth = 2, alpha=0.8,label = 'model mean seaice')
x_fill = np.concatenate([year2,np.flipud(year2)])
y_fill = np.concatenate([multimodel_min,np.flipud(multimodel_max)])
ax1.fill_between(x_fill,y_fill,edgecolor = 'none', facecolor='blue', alpha=0.2)

ax2 = ax1.twinx()
loc = np.where((reynolds_data[:,0] <= 1850) & (reynolds_data[:,0] >= 850) )[0]
y_reynolds = reynolds_data[loc[0]:loc[-1],1]
lab2, = ax2.plot(reynolds_data[loc[0]:loc[-1],0],running_mean.running_mean(signal.detrend(y_reynolds),1),'r',linewidth = 2, alpha=0.5,label = 'd18O')

ax1.set_ylabel('seaice')
ax2.set_ylabel('d18OT')
ax1.set_xlim([900,1900])
ax1.set_ylim([1.5,-1.5])
ax1.plot([900,1900],[0,0],'k')


ax1.set_xlabel('year')

ax1.legend([lab1],prop={'size':10},loc=1).draw_frame(False)
ax2.legend([lab2],prop={'size':10},loc=4).draw_frame(False)
plt.savefig('/home/ph290/Documents/figures/multi_model_mean_seaice.png')
plt.show(block = False)


'''
#spatial anlayis
'''


lon_west2 = -60
lon_east2 = +70
lat_south2 = 30.0
lat_north2 = 90.0

region2 = iris.Constraint(longitude=lambda v: lon_west2 <= v <= lon_east2,latitude=lambda v: lat_south2 <= v <= lat_north2)

mean_cube2 = mean_cube.copy()
mean_cube2.data = np.roll(mean_cube.data,180,axis = 2)
mean_cube2.coord('longitude').points = np.linspace(-180,180,360)

mean_cube2 = mean_cube2.extract(region2)

mca_cube = mean_cube2[0:250]
lia_cube = mean_cube2[400:700]

y_mca = y[0:250]
loc1 = np.where(y_mca > stats.nanmean(y))
loc2 = np.where(y_mca < stats.nanmean(y))

mca_cube_diff = mca_cube[loc1].collapsed('time',iris.analysis.MEAN)-mca_cube[loc2].collapsed('time',iris.analysis.MEAN)

y_lia = y[400:700]
loc1b = np.where(y_lia > stats.nanmean(y))
loc2b = np.where(y_lia < stats.nanmean(y))

lia_cube_diff = lia_cube[loc1b].collapsed('time',iris.analysis.MEAN)-lia_cube[loc2b].collapsed('time',iris.analysis.MEAN)

plt.close('all')
plt.figure(1)
qplt.contourf(mca_cube_diff,np.linspace(-15,15,31))
plt.gca().coastlines()
plt.title('high/low seaice in MCA')
plt.savefig('/home/ph290/Documents/figures/multi_model_mean_seaice_dmca.png')
#plt.show(block = False)


plt.figure(2)
qplt.contourf(lia_cube_diff,np.linspace(-15,15,31))
plt.gca().coastlines()
plt.title('high/low seaice in LIA')
plt.savefig('/home/ph290/Documents/figures/multi_model_mean_seaice_dlia.png')
#plt.show(block = False)


np.savetxt('/home/ph290/Documents/figures/seaice_cmip5.txt', np.c_[year2,y])

'''
#precipitation
'''

lon_west3 = -20+360
lon_east3 = -15+360
lat_south3 = 66
lat_north3 = 70

region3 = iris.Constraint(longitude=lambda v: lon_west3 <= v <= lon_east3,latitude=lambda v: lat_south3 <= v <= lat_north3)

modelsb_pr = model_names(input_directory2,variables[1])

cube1_pr = iris.load_cube(input_directory2+modelsb[0]+'*'+variables[1]+'*.nc')[0]

modelbs2_pr = [] 
cubes_n_hem_pr = []
ts_region_pr = []

for model in modelsb_pr:
        print 'processing: '+model
        try:
                cube = iris.load_cube(input_directory2+model+'*'+variables[1]+'*.nc')
        except:
                cube = iris.load(input_directory2+model+'*'+variables[1]+'*.nc')
                cube = cube[0]
        tmp1 = cube.extract(region1)
        cubes_n_hem_pr.append(tmp1)
        #qplt.contourf(tmp1[0],50)
        #plt.show()
	tmp3 = cube.extract(region3)
        ts_region_pr.append(tmp3.collapsed(['latitude','longitude'],iris.analysis.MEAN))

data = np.zeros([1000,np.size(ts_region_pr)])
j=0
for i,ts in enumerate(ts_region_pr):
        print modelsb_pr[i]
        data[:,j] = signal.detrend(ts.data[0:1000])
        j += 1

multimodel_mean_pr = data.mean(axis = 1)
multimodel_max_pr = data.max(axis = 1)
multimodel_min_pr = data.min(axis = 1)



plt.close('all')
fig = plt.subplots(2,1,figsize=(16, 7))

#plt 1
ax1 = plt.subplot(2,1,1)
year2 = year[0]+np.arange(1000)
y = running_mean.running_mean(multimodel_mean_pr-np.mean(multimodel_mean_pr),1)
ax1.plot(year2,y,'g',linewidth = 2, alpha=0.8,label = 'model mean precipitation')
x_fill = np.concatenate([year2,np.flipud(year2)])
y_fill = np.concatenate([multimodel_min_pr,np.flipud(multimodel_max_pr)])
ax1.fill_between(x_fill,y_fill,edgecolor = 'none', facecolor='green', alpha=0.2)

ax2 = ax1.twinx()
loc = np.where((reynolds_data[:,0] <= 1850) & (reynolds_data[:,0] >= 850) )[0]
y_reynolds = reynolds_data[loc[0]:loc[-1],1]
ax2.plot(reynolds_data[loc[0]:loc[-1],0],running_mean.running_mean(signal.detrend(y_reynolds),1),'r',linewidth = 2, alpha=0.5,label = 'd18O')

ax1.set_ylabel('precip')
ax2.set_ylabel('d18OT')
ax1.set_xlim([900,1900])
ax1.set_ylim([0.000008,-0.000008])
ax1.plot([900,1900],[0,0],'k')


#plt2
ax1 = plt.subplot(2,1,2)
year2 = year[0]+np.arange(1000)
y = running_mean.running_mean(multimodel_mean_pr-np.mean(multimodel_mean_pr),1)
ax1.plot(year2,y,'g',linewidth = 2, alpha=0.8,label = 'model mean precipitation')
x_fill = np.concatenate([year2,np.flipud(year2)])
y_fill = np.concatenate([multimodel_min_pr,np.flipud(multimodel_max_pr)])
ax1.fill_between(x_fill,y_fill,edgecolor = 'none', facecolor='green', alpha=0.2)

ax2 = ax1.twinx()
loc = np.where((reynolds_data[:,0] <= 1850) & (reynolds_data[:,0] >= 850) )[0]
y_reynolds = reynolds_data[loc[0]:loc[-1],1]
ax2.plot(reynolds_data[loc[0]:loc[-1],0],running_mean.running_mean(signal.detrend(y_reynolds),1),'r',linewidth = 2, alpha=0.5,label = 'd18O')

ax1.set_ylabel('precip')
ax2.set_ylabel('d18OT')
ax1.set_xlim([900,1900])
ax1.set_ylim([-0.000008,0.000008])
ax1.plot([900,1900],[0,0],'k')

plt.savefig('/home/ph290/Documents/figures/multi_model_mean_precip.png')
#plt.show()


'''
'''
precipitation, spatial
'''

tmp = np.shape(cubes_n_hem_pr[0].data)
tmp_data = np.ma.empty([np.size(cubes_n_hem),1000,tmp[1],tmp[2]])

for i in np.arange(np.size(cubes_n_hem)):
        tmp_data[i] = cubes_n_hem[i].data[0:1000]


mean_cube = cubes_n_hem[0].copy()
mean_cube.data = np.mean(tmp_data,axis = 0)


mca_cube = mean_cube2[0:250]
lia_cube = mean_cube2[400:700]

y_mca = y[0:250]
loc1 = np.where(y_mca > stats.nanmean(y))
loc2 = np.where(y_mca < stats.nanmean(y))

mca_cube_diff = mca_cube[loc1].collapsed('time',iris.analysis.MEAN)-mca_cube[loc2].collapsed('time',iris.analysis.MEAN)

y_lia = y[400:700]
loc1b = np.where(y_lia > stats.nanmean(y))
loc2b = np.where(y_lia < stats.nanmean(y))

lia_cube_diff = lia_cube[loc1b].collapsed('time',iris.analysis.MEAN)-lia_cube[loc2b].collapsed('time',iris.analysis.MEAN)

plt.close('all')
plt.figure(1)
qplt.contourf(mca_cube_diff,np.linspace(-5,5,31))
plt.gca().coastlines()
plt.title('high/low pr in MCA')
plt.savefig('/home/ph290/Documents/figures/multi_model_mean_pr_dmca.png')
#plt.show(block = False)


plt.figure(2)
qplt.contourf(lia_cube_diff,np.linspace(-5,5,31))
plt.gca().coastlines()
plt.title('high/low pr in LIA')
plt.savefig('/home/ph290/Documents/figures/multi_model_mean_pr_dlia.png')
#plt.show(block = False)


