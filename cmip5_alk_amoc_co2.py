# import numpy as np
# import iris
# import matplotlib.pyplot as plt
# import iris.coord_categorisation
# import iris.analysis
# import iris.analysis.cartography
# import ols
# from running_mean import * 
# import cartopy
# import glob
# import subprocess
# import os
# 
# def my_callback(cube, field,files_tmp):
#     cube.attributes.pop('history')
#     cube.attributes.pop('tracking_id')
#     cube.attributes.pop('creation_date')
#     return cube
#     
# def model_name(files):
# 	models = []
# 	for file in files:
# 		models.append(file.split('/')[-1].split('_')[2])
# 	return np.unique(models)
# 
# amoc_dir = '/home/ph290/data0/cmip5_data/msftmyz/piControl/'
# talk_dir = '/data/data0/ph290/cmip5_data/talk_mon/'
# fgco2_dir = '/data/data0/ph290/cmip5_data/picontrol/fgco2/'
# tos_dir = '/data/data0/ph290/cmip5_data/tos/piControl/'
# 
# amoc_models = model_name(glob.glob(amoc_dir+'/*.nc'))
# talk_models = model_name(glob.glob(talk_dir+'/*.nc'))
# fgco2_models = model_name(glob.glob(fgco2_dir+'/*.nc'))
# tos_models = model_name(glob.glob(tos_dir+'/*.nc'))
# tmp = np.intersect1d(amoc_models,talk_models)
# tmp = np.intersect1d(fgco2_models,tmp)
# models_uniqiue = np.intersect1d(tos_models,tmp)
# # array(['CESM1-BGC', 'CNRM-CM5', 'CNRM-CM5-2', 'CanESM2', 'MPI-ESM-LR','MPI-ESM-MR', 'NorESM1-ME'], 
# 
# vars = ['talk','fgco2','tos']
# dirs = [talk_dir,fgco2_dir,tos_dir]
# output_directory = "/home/ph290/data0/cmip5_data/regridded/"
# 
# #concatenate and regrid file
# for i,var in enumerate(vars):
# 	for model in models_uniqiue:
# 		test = glob.glob(output_directory+var+'_'+model+'_regridded.nc')
# 		if len(test) == 0:
# 			files = glob.glob(dirs[i]+'/*'+model+'*.nc')
# 			files = ' '.join(files)
# 			subprocess.call(['cdo mergetime '+files+' /home/ph290/data0/tmp/temp.nc'], shell=True)
# 			subprocess.call(['cdo remapbil,r360x180 -selname,'+var+' /home/ph290/data0/tmp/temp.nc '+output_directory+var+'_'+model+'_regridded.nc'], shell=True)
# 			subprocess.call('rm /home/ph290/data0/tmp/temp.nc', shell=True)
# 
# var = 'msftmyz'
# for model in models_uniqiue:
# 	test = glob.glob(output_directory+var+'_'+model+'_regridded.nc')
# 	if len(test) == 0:
# 		files = glob.glob(amoc_dir+'/*'+model+'*.nc')
# 		files = ' '.join(files)
# 		subprocess.call('cdo mergetime '+files+' '+output_directory+var+'_'+model+'_regridded.nc', shell=True)
# 
# model = 'CESM1-BGC'
# 
# def calculate_stuff(model):
# 	print model
# 	amoc_file = output_directory+'msftmyz_'+model+'_regridded.nc'
# 	talk_file = output_directory+'talk_'+model+'_regridded.nc'
# 	fgco2_file = output_directory+'fgco2_'+model+'_regridded.nc'
# 	tos_file = output_directory+'tos_'+model+'_regridded.nc'
# 
# 	print 'processing AMOC'
# 	amoc_mon = iris.load_cube(amoc_file)
# 	# 	amoc_mon = iris.cube.CubeList.concatenate(iris.load(amoc_file,callback = my_callback))
# 	iris.coord_categorisation.add_year(amoc_mon, 'time', name='year')
# 	amoc = amoc_mon.aggregated_by('year', iris.analysis.MEAN)
# 
# 	try:
# 		lats = amoc.coord('latitude').points
# 	except iris.exceptions.CoordinateNotFoundError:
# 		lats = amoc.coord('grid_latitude').points
# 	lat = np.where(lats >= 26)[0][0]
# 	amoc_strength = np.max(amoc.data[:,0,:,lat],axis = 1)
# 
# 	print 'processing talk'
# 	talk_mon = iris.load_cube(talk_file)
# # 	talk_mon =  iris.cube.CubeList.concatenate(iris.load(talk_file,callback = my_callback))
# 	iris.coord_categorisation.add_year(talk_mon, 'time', name='year')
# 	talk = talk_mon.aggregated_by('year', iris.analysis.MEAN)
# # 	constraint = iris.Constraint(depth = 0)
# # 	talk = talk.extract(constraint)
# 
# 	print 'processing fgco2'
# 	fgco2_mon = iris.load_cube(fgco2_file)
# # 	fgco2_mon =  iris.cube.CubeList.concatenate(iris.load(fgco2_file,callback = my_callback))
# 	iris.coord_categorisation.add_year(fgco2_mon, 'time', name='year')
# 	fgco2 = fgco2_mon.aggregated_by('year', iris.analysis.MEAN)
# 
# 	print 'processing tos'
# 	tos_mon = iris.load_cube(tos_file)
# # 	tos_mon =  iris.cube.CubeList.concatenate(iris.load(tos_file,callback = my_callback))
# 	iris.coord_categorisation.add_year(tos_mon, 'time', name='year')
# 	tos = tos_mon.aggregated_by('year', iris.analysis.MEAN)
# 
# 	print 'processing regions'
# 	lon_west = 360-80
# 	lon_east = 360
# 	lat_south = 26.0
# 	lat_north = 70
# 
# 	lat_south2 = 0.0
# 	lat_north2 = 26
# 
# 	region = iris.Constraint(longitude=lambda v: lon_west <= v <= lon_east,latitude=lambda v: lat_south <= v <= lat_north)
# 	region2 = iris.Constraint(longitude=lambda v: lon_west <= v <= lon_east,latitude=lambda v: lat_south2 <= v <= lat_north2)
# 
# 	talk_region = talk.extract(region)
# 	try:
# 		talk_region.coord('latitude').guess_bounds()
# 		talk_region.coord('longitude').guess_bounds()
# 	except ValueError:
# 		'already has bounds'
# 	grid_areas = iris.analysis.cartography.area_weights(talk_region)
# 	talk_ts = talk_region.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas)
# 
# 	talk_region2 = talk.extract(region2)
# 	try:
# 		talk_region2.coord('latitude').guess_bounds()
# 		talk_region2.coord('longitude').guess_bounds()
# 	except ValueError:
# 		'already has bounds'
# 	grid_areas2 = iris.analysis.cartography.area_weights(talk_region2)
# 	talk_ts2 = talk_region2.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas2)
# 
# 
# 	fgco2_region = fgco2.extract(region)
# 	try:
# 		fgco2_region.coord('latitude').guess_bounds()
# 		fgco2_region.coord('longitude').guess_bounds()
# 	except ValueError:
# 		'already has bounds'
# 	grid_areas3 = iris.analysis.cartography.area_weights(fgco2_region)
# 	fgco2_ts = fgco2_region.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas3)
# 
# 	tos_region = tos.extract(region)
# 	try:
# 		tos_region.coord('latitude').guess_bounds()
# 		tos_region.coord('longitude').guess_bounds()
# 	except ValueError:
# 		'already has bounds'
# 	grid_areas4 = iris.analysis.cartography.area_weights(tos_region)
# 	tos_ts = tos_region.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights=grid_areas4)
# 
# 	return amoc_strength,fgco2_ts,talk_ts,tos_ts,talk_ts2
# 
# 
# CESM1BGC_amoc_strength,CESM1BGC_fgco2_ts,CESM1BGC_talk_ts,CESM1BGC_tos_ts,CESM1BGC_talk_ts2 = calculate_stuff('CESM1-BGC')
# CNRMCM5_amoc_strength,CNRMCM5_fgco2_ts,CNRMCM5_talk_ts,CNRMCM5_tos_ts,CNRMCM5_talk_ts2 = calculate_stuff('CNRM-CM5')
# CNRMCM52_amoc_strength,CNRMCM52_fgco2_ts,CNRMCM52_talk_ts,CNRMCM52_tos_ts,CNRMCM52_talk_ts2 = calculate_stuff('CNRM-CM5-2')
# CanESM2_amoc_strength,CanESM2_fgco2_ts,CanESM2_talk_ts,CanESM2_tos_ts,CanESM2_talk_ts2 = calculate_stuff('CanESM2')
# MPIESMLR_amoc_strength,MPIESMLR_fgco2_ts,MPIESMLR_talk_ts,MPIESMLR_tos_ts,MPIESMLR_talk_ts2 = calculate_stuff('MPI-ESM-LR')
# MPIESMMR_amoc_strength,MPIESMMR_fgco2_ts,MPIESMMR_talk_ts,MPIESMMR_tos_ts,MPIESMMR_talk_ts2 = calculate_stuff('MPI-ESM-MR')
# NorESM1ME_amoc_strength,NorESM1ME_fgco2_ts,NorESM1ME_talk_ts,NorESM1ME_tos_ts,NorESM1ME_talk_ts2 = calculate_stuff('NorESM1-ME')

save_stuff = [CESM1BGC_amoc_strength,CESM1BGC_fgco2_ts,CESM1BGC_talk_ts,CESM1BGC_tos_ts,CESM1BGC_talk_ts2,CNRMCM5_amoc_strength,CNRMCM5_fgco2_ts,CNRMCM5_talk_ts,CNRMCM5_tos_ts,CNRMCM5_talk_ts2,CNRMCM52_amoc_strength,CNRMCM52_fgco2_ts,CNRMCM52_talk_ts,CNRMCM52_tos_ts,CNRMCM52_talk_ts2,CanESM2_amoc_strength,CanESM2_fgco2_ts,CanESM2_talk_ts,CanESM2_tos_ts,CanESM2_talk_ts2,MPIESMLR_amoc_strength,MPIESMLR_fgco2_ts,MPIESMLR_talk_ts,MPIESMLR_tos_ts,MPIESMLR_talk_ts2,MPIESMMR_amoc_strength,MPIESMMR_fgco2_ts,MPIESMMR_talk_ts,MPIESMMR_tos_ts,MPIESMMR_talk_ts2,NorESM1ME_amoc_strength,NorESM1ME_fgco2_ts,NorESM1ME_talk_ts,NorESM1ME_tos_ts,NorESM1ME_talk_ts2]

'''
save variables when done
'''


import pickle

f = open('store.pckl', 'w')
pickle.dump(save_stuff, f)
f.close()

'''
restore with
'''
# import pickle
# f = open('store.pckl')
# save_stuff = pickle.load(f)
# f.close()


# coord = amoc.coord('time')
# dt = coord.units.num2date(coord.points)
# amoc_year = np.array([coord.units.num2date(value).year for value in coord.points])
# 
# coord = talk.coord('time')
# dt = coord.units.num2date(coord.points)
# talk_year = np.array([coord.units.num2date(value).year for value in coord.points])

# 
# # amoc_strength2 = amoc_strength[:-10]
# # 
# # averaging = 20
# # 
# # x = np.empty([amoc_strength2.size-averaging,3])
# # x[:,0] = running_mean(amoc_strength2,averaging)[:-1*averaging]
# # x[:,1] = running_mean(talk_ts.data*1.0e10,averaging)[:-averaging]
# # #x[:,2] = running_mean(tos_ts.data,averaging)[:-averaging]
# # x[:,2] = running_mean(talk_ts2.data,averaging)[:-averaging]
# # 
# # y = running_mean(fgco2_ts.data*1.0e9,averaging)[:-averaging]
# # 
# # mymodel = ols.ols(y,x,'y',['x1','x2','x3'])
# # 
# # plt.scatter(y,mymodel.b[0]+mymodel.b[1]*x[:,0]+mymodel.b[2]*x[:,1]+mymodel.b[3]*x[:,2])
# # plt.show()
# 
# 
