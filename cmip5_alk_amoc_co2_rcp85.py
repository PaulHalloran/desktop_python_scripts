import numpy as np
import iris
import matplotlib.pyplot as plt
import iris.coord_categorisation
import iris.analysis
import iris.analysis.cartography
import ols
from running_mean import * 
import cartopy
import glob
import subprocess
import os
from scipy.stats.mstats import *
import matplotlib as mpl

def my_callback(cube, field,files_tmp):
    cube.attributes.pop('history')
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    return cube
    
def model_name(files):
	models = []
	for file in files:
		models.append(file.split('/')[-1].split('_')[2])
	return np.unique(models)

amoc_dir = '/media/usb_external1/cmip5/rcp85_data_rapid/msftmyz/'
talk_dir = '/media/usb_external1/cmip5/rcp85_data_rapid/talk/'
fgco2_dir = '/media/usb_external1/cmip5/rcp85_data_rapid/fgco2/'
tos_dir = '/media/usb_external1/cmip5/rcp85_data_rapid/tos/'

amoc_models = model_name(glob.glob(amoc_dir+'/*.nc'))
talk_models = model_name(glob.glob(talk_dir+'/*.nc'))
fgco2_models = model_name(glob.glob(fgco2_dir+'/*.nc'))
tos_models = model_name(glob.glob(tos_dir+'/*.nc'))
tmp = np.intersect1d(amoc_models,talk_models)
tmp = np.intersect1d(fgco2_models,tmp)
models_uniqiue = np.intersect1d(tos_models,tmp)
# array(['CESM1-BGC', 'CNRM-CM5', 'CNRM-CM5-2', 'CanESM2', 'MPI-ESM-LR','MPI-ESM-MR', 'NorESM1-ME'], 

vars = ['talk','fgco2','tos']
dirs = [talk_dir,fgco2_dir,tos_dir]
output_directory = "/media/usb_external1/cmip5/rcp85_data_rapid/regridded/"

#concatenate and regrid file
for i,var in enumerate(vars):
	for model in models_uniqiue:
		test = glob.glob(output_directory+var+'_'+model+'_regridded.nc')
		if len(test) == 0:
			files = glob.glob(dirs[i]+'/*'+model+'*.nc')
			files = ' '.join(files)
			subprocess.call(['cdo mergetime '+files+' /home/ph290/data0/tmp/temp.nc'], shell=True)
			subprocess.call(['cdo remapbil,r360x180 -selname,'+var+' /home/ph290/data0/tmp/temp.nc '+output_directory+var+'_'+model+'_regridded.nc'], shell=True)
			subprocess.call('rm /home/ph290/data0/tmp/temp.nc', shell=True)

var = 'msftmyz'
for model in models_uniqiue:
	test = glob.glob(output_directory+var+'_'+model+'_regridded.nc')
	if len(test) == 0:
		files = glob.glob(amoc_dir+'/*'+model+'*.nc')
		files = ' '.join(files)
		subprocess.call('cdo mergetime '+files+' '+output_directory+var+'_'+model+'_regridded.nc', shell=True)


def calculate_stuff(model):
	print model
	amoc_file = output_directory+'msftmyz_'+model+'_regridded.nc'
	talk_file = output_directory+'talk_'+model+'_regridded.nc'
	fgco2_file = output_directory+'fgco2_'+model+'_regridded.nc'
	tos_file = output_directory+'tos_'+model+'_regridded.nc'

	print 'processing AMOC'
	amoc_mon = iris.load_cube(amoc_file)
	# 	amoc_mon = iris.cube.CubeList.concatenate(iris.load(amoc_file,callback = my_callback))
	iris.coord_categorisation.add_year(amoc_mon, 'time', name='year')
	amoc = amoc_mon.aggregated_by('year', iris.analysis.MEAN)

	try:
		lats = amoc.coord('latitude').points
	except iris.exceptions.CoordinateNotFoundError:
		lats = amoc.coord('grid_latitude').points
	lat = np.where(lats >= 26)[0][0]
	amoc_strength = np.max(amoc.data[:,0,:,lat],axis = 1)

	print 'processing talk'
	talk_mon = iris.load_cube(talk_file)
# 	talk_mon =  iris.cube.CubeList.concatenate(iris.load(talk_file,callback = my_callback))
	iris.coord_categorisation.add_year(talk_mon, 'time', name='year')
	talk = talk_mon.aggregated_by('year', iris.analysis.MEAN)
# 	constraint = iris.Constraint(depth = 0)
# 	talk = talk.extract(constraint)

	print 'processing fgco2'
	fgco2_mon = iris.load_cube(fgco2_file)
# 	fgco2_mon =  iris.cube.CubeList.concatenate(iris.load(fgco2_file,callback = my_callback))
	iris.coord_categorisation.add_year(fgco2_mon, 'time', name='year')
	fgco2 = fgco2_mon.aggregated_by('year', iris.analysis.MEAN)

	print 'processing tos'
	tos_mon = iris.load_cube(tos_file)
# 	tos_mon =  iris.cube.CubeList.concatenate(iris.load(tos_file,callback = my_callback))
	iris.coord_categorisation.add_year(tos_mon, 'time', name='year')
	tos = tos_mon.aggregated_by('year', iris.analysis.MEAN)

	print 'processing regions'
	lon_west = 360-80
	lon_east = 360
	lat_south = 26.0
	lat_north = 70

	lat_south2 = 0.0
	lat_north2 = 26

	region = iris.Constraint(longitude=lambda v: lon_west <= v <= lon_east,latitude=lambda v: lat_south <= v <= lat_north)
	region2 = iris.Constraint(longitude=lambda v: lon_west <= v <= lon_east,latitude=lambda v: lat_south2 <= v <= lat_north2)

	talk_region = talk.extract(region)
	try:
		talk_region.coord('latitude').guess_bounds()
		talk_region.coord('longitude').guess_bounds()
	except ValueError:
		'already has bounds'
	grid_areas = iris.analysis.cartography.area_weights(talk_region)
	talk_ts = talk_region.collapsed(['latitude','longitude'],iris.analysis.SUM, weights=grid_areas)

	talk_region2 = talk.extract(region2)
	try:
		talk_region2.coord('latitude').guess_bounds()
		talk_region2.coord('longitude').guess_bounds()
	except ValueError:
		'already has bounds'
	grid_areas2 = iris.analysis.cartography.area_weights(talk_region2)
	talk_ts2 = talk_region2.collapsed(['latitude','longitude'],iris.analysis.SUM, weights=grid_areas2)


	fgco2_region = fgco2.extract(region)
	try:
		fgco2_region.coord('latitude').guess_bounds()
		fgco2_region.coord('longitude').guess_bounds()
	except ValueError:
		'already has bounds'
	grid_areas3 = iris.analysis.cartography.area_weights(fgco2_region)
	fgco2_ts = fgco2_region.collapsed(['latitude','longitude'],iris.analysis.SUM, weights=grid_areas3)

	tos_region = tos.extract(region)
	try:
		tos_region.coord('latitude').guess_bounds()
		tos_region.coord('longitude').guess_bounds()
	except ValueError:
		'already has bounds'
	grid_areas4 = iris.analysis.cartography.area_weights(tos_region)
	tos_ts = tos_region.collapsed(['latitude','longitude'],iris.analysis.SUM, weights=grid_areas4)

	return amoc_strength,fgco2_ts,talk_ts,tos_ts,talk_ts2

'MPI-ESM-LR', 'MPI-ESM-MR', 'NorESM1-ME'

# CESM1BGC_amoc_strength,CESM1BGC_fgco2_ts,CESM1BGC_talk_ts,CESM1BGC_tos_ts,CESM1BGC_talk_ts2 = calculate_stuff('CESM1-BGC')
# CNRMCM5_amoc_strength,CNRMCM5_fgco2_ts,CNRMCM5_talk_ts,CNRMCM5_tos_ts,CNRMCM5_talk_ts2 = calculate_stuff('CNRM-CM5')
# CNRMCM52_amoc_strength,CNRMCM52_fgco2_ts,CNRMCM52_talk_ts,CNRMCM52_tos_ts,CNRMCM52_talk_ts2 = calculate_stuff('CNRM-CM5-2')

#CanESM2_amoc_strength,CanESM2_fgco2_ts,CanESM2_talk_ts,CanESM2_tos_ts,CanESM2_talk_ts2 = calculate_stuff('CanESM2')

 #MPIESMLR_amoc_strength,MPIESMLR_fgco2_ts,MPIESMLR_talk_ts,MPIESMLR_tos_ts,MPIESMLR_talk_ts2 = calculate_stuff('MPI-ESM-LR')
 #MPIESMMR_amoc_strength,MPIESMMR_fgco2_ts,MPIESMMR_talk_ts,MPIESMMR_tos_ts,MPIESMMR_talk_ts2 = calculate_stuff('MPI-ESM-MR')
 #NorESM1ME_amoc_strength,NorESM1ME_fgco2_ts,NorESM1ME_talk_ts,NorESM1ME_tos_ts,NorESM1ME_talk_ts2 = calculate_stuff('NorESM1-ME')

modela = np.array([CanESM2_amoc_strength,CanESM2_fgco2_ts,CanESM2_talk_ts,CanESM2_tos_ts,CanESM2_talk_ts2])
modelb = np.array([MPIESMMR_amoc_strength,MPIESMMR_fgco2_ts,MPIESMMR_talk_ts,MPIESMMR_tos_ts,MPIESMMR_talk_ts2])
modelc = np.array([NorESM1ME_amoc_strength,NorESM1ME_fgco2_ts,NorESM1ME_talk_ts,NorESM1ME_tos_ts,NorESM1ME_talk_ts2])

all_model_results = np.array([modela,modelb,modelc])

# save_stuff = [CESM1BGC_amoc_strength,CESM1BGC_fgco2_ts,CESM1BGC_talk_ts,CESM1BGC_tos_ts,CESM1BGC_talk_ts2,CNRMCM5_amoc_strength,CNRMCM5_fgco2_ts,CNRMCM5_talk_ts,CNRMCM5_tos_ts,CNRMCM5_talk_ts2,CNRMCM52_amoc_strength,CNRMCM52_fgco2_ts,CNRMCM52_talk_ts,CNRMCM52_tos_ts,CNRMCM52_talk_ts2,CanESM2_amoc_strength,CanESM2_fgco2_ts,CanESM2_talk_ts,CanESM2_tos_ts,CanESM2_talk_ts2,MPIESMLR_amoc_strength,MPIESMLR_fgco2_ts,MPIESMLR_talk_ts,MPIESMLR_tos_ts,MPIESMLR_talk_ts2,MPIESMMR_amoc_strength,MPIESMMR_fgco2_ts,MPIESMMR_talk_ts,MPIESMMR_tos_ts,MPIESMMR_talk_ts2,NorESM1ME_amoc_strength,NorESM1ME_fgco2_ts,NorESM1ME_talk_ts,NorESM1ME_tos_ts,NorESM1ME_talk_ts2]
# 
# '''
# save variables when done
# '''
# 
# 
# import pickle
# 
# f = open('store.pckl', 'w')
# pickle.dump(save_stuff, f)
# f.close()
# 
# '''
# restore with
# '''
# import pickle
# f = open('store.pckl')
# save_stuff = pickle.load(f)
# f.close()


coord = CanESM2_fgco2_ts.coord('time')
dt = coord.units.num2date(coord.points)
fgco2_year = np.array([coord.units.num2date(value).year for value in coord.points])
# 
# coord = talk.coord('time')
# dt = coord.units.num2date(coord.points)
# talk_year = np.array([coord.units.num2date(value).year for value in coord.points])

co2_file = '/home/ph290/data0/misc_data/RCP85_MIDYR_CONC.DAT'
co2_data = np.genfromtxt(co2_file,skip_header = 40)
loc = np.where((co2_data[:,0] >= 2006) & (co2_data[:,0] <= 2100))
co2_yr = co2_data[loc[0],0]
co2_values = co2_data[loc[0],3]

amoc_strength = []
fgco2_ts =  []
talk_ts = []
talk_ts2 = []
co2_values2 = []
tos_ts = []

averaging = 5.0

for i in range(all_model_results.shape[0]):
    amoc_strength = np.concatenate([amoc_strength,running_mean(all_model_results[i,0]/np.mean(all_model_results[i,0]),averaging)[0:-averaging]])
    fgco2_ts = np.concatenate([fgco2_ts,running_mean(all_model_results[i,1].data/np.mean(all_model_results[i,1].data),averaging)[0:-averaging]])
    talk_ts = np.concatenate([talk_ts,running_mean(all_model_results[i,2].data/np.mean(all_model_results[i,2].data),averaging)[0:-averaging]])
    tos_ts = np.concatenate([tos_ts,running_mean(all_model_results[i,3].data/np.mean(all_model_results[i,3].data),averaging)[0:-averaging]])
    talk_ts2 = np.concatenate([talk_ts2,running_mean(all_model_results[i,4].data/np.mean(all_model_results[i,4].data),averaging)[0:-averaging]])
    co2_values2 = np.concatenate([co2_values2,running_mean(co2_values/np.mean(co2_values),averaging)[0:-averaging]])

xs = [amoc_strength,talk_ts,co2_values2,talk_ts2,tos_ts]

x = np.empty([amoc_strength.size,5])
x[:,0] = xs[0]
x[:,1] = xs[1]
x[:,2] = xs[2]
x[:,3] = xs[3]
x[:,4] = xs[4]
y = fgco2_ts
#*(60.0*60.0*24.0*365.0)/1.0e12
mymodel = ols.ols(y,x,'y',['x1','x2','x3','x4','x5'])

x2 = np.empty([amoc_strength.size,1])
x2[:,0] = xs[2]
mymodel2 = ols.ols(y,x2,'y',['x1'])

x3 = np.empty([amoc_strength.size,2])
x3[:,0] = xs[2]
x3[:,1] = xs[0]
mymodel3 = ols.ols(y,x3,'y',['x1','x2'])

x4 = np.empty([amoc_strength.size,4])
x4[:,0] = xs[2]
x4[:,1] = xs[0]
x4[:,2] = xs[1]
x4[:,3] = xs[3]
mymodel4 = ols.ols(y,x4,'y',['x1','x2','x3','x4'])

x5 = np.empty([amoc_strength.size,5])
x5[:,0] = xs[2]
x5[:,1] = xs[0]
x5[:,2] = xs[1]
x5[:,3] = xs[3]
x5[:,4] = xs[4]
mymodel5 = ols.ols(y,x5,'y',['x1','x2','x3','x4','x5'])

# plt.scatter(y,mymodel.b[0]+mymodel.b[1]*x[:,0]+mymodel.b[2]*x[:,1]+mymodel.b[3]*x[:,2]+mymodel.b[4]*x[:,3]+mymodel.b[5]*x[:,4],color='yellow')
xa = mymodel2.b[0]+mymodel2.b[1]*x2[:,0]
slope_a, intercept_a, r_value_a, p_value_a, std_err_a = linregress(y,xa)
xb = mymodel3.b[0]+mymodel3.b[1]*x3[:,0]+mymodel3.b[2]*x3[:,1]
slope_b, intercept_b, r_value_b, p_value_b, std_err_b = linregress(y,xb)
xc = mymodel4.b[0]+mymodel4.b[1]*x4[:,0]+mymodel4.b[2]*x4[:,1]+mymodel4.b[3]*x4[:,2]+mymodel4.b[4]*x4[:,3]
slope_c, intercept_c, r_value_c, p_value_c, std_err_c = linregress(y,xc)
xd = mymodel5.b[0]+mymodel5.b[1]*x5[:,0]+mymodel5.b[2]*x5[:,1]+mymodel5.b[3]*x5[:,2]+mymodel5.b[4]*x5[:,3]+mymodel5.b[5]*x5[:,4]
slope_d, intercept_d, r_value_d, p_value_d, std_err_d = linregress(y,xd)
values = np.linspace(0.7,1.25)

font = {'family' : 'normal',
'weight' : 'bold',
'size' : 16}

mpl.rc('font', **font)

fig, ax = plt.subplots()
ax.scatter(y,xa,color='grey',marker='D',facecolors='none', label='prediction using atm. CO$_2$ only')
ax.plot(values,slope_a*values+intercept_a,color='grey',linewidth = 3)
#ax.scatter(y,xb,color='green',marker='o',facecolors='none', label='prediction using atm. CO$_2$ + AMOC')
#ax.plot(values,slope_b*values+intercept_b,color='green')
ax.scatter(y,xc,color='k',marker='o',facecolors='none', label='prediction using atm. CO$_2$ + AMOC + Alkalinity')
ax.plot(values,slope_c*values+intercept_c,color='k', label='least squares best fit',linewidth = 3)
#ax.scatter(y,xd,color='grey',marker='o',facecolors='none', label='prediction using atm. CO$_2$ + AMOC + Alkalinity * T')
#ax.plot(values,slope_d*values+intercept_c,color='grey', label='least squares best fit')
ax.set_xlim(0.6,1.3)
ax.set_ylim(0.6,1.3)
ax.set_xlabel('ESM simulated N. Atlantic CO$_2$ flux\n(normalised)')
ax.set_ylabel('statistically predicted N. Atlantic CO$_2$ flux\n(normalised)')
legend = ax.legend(loc='lower right',prop={'size':11}).draw_frame(False)
plt.tight_layout()
ax.set_aspect('equal')
plt.savefig('/home/ph290/Documents/figures/cmip5_alk_amoc_co2.pdf') 
#plt.show()

