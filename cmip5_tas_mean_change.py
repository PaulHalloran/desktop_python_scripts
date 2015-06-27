# import numpy as np
# import matplotlib.pyplot as plt
# import iris
# import glob
# import iris.experimental.concatenate
# import iris.analysis
# import iris.quickplot as qplt
# import iris.analysis.cartography
# import cartopy.crs as ccrs
# import subprocess
# from iris.coords import DimCoord
# import iris.coord_categorisation
# import matplotlib as mpl

# def my_callback(cube, field,files_tmp):
#     # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
#     cube.attributes.pop('history')
#     cube.attributes.pop('tracking_id')
#     cube.attributes.pop('creation_date')
#     #if np.size(cube) > 1:
#     #cube = iris.experimental.concatenate.concatenate(cube)
#     return cube

# def monthly_to_yearly(cube):
#     #if np.size(cube._aux_coords_and_dims) < 2:
#     iris.coord_categorisation.add_year(cube, 'time', name='year2')
#     cube_tmp = cube.aggregated_by('year2', iris.analysis.MEAN)
#     return cube_tmp


# directory = '/home/ph290/data1/cmip5_data/tas_and_pr/'

# run_names = ['historical','rcp26','rcp45','rcp60','rcp85']

# model_names_final_all_models = []
# all_cubes_all_models = []
# all_cubes_timeseries_all_models = []
# all_cubes_timeseries_year_all_models = []

# for run_name in run_names:

#     print 'processing '+run_name

#     files=glob.glob(directory+run_name+'/tas_Amon*.nc')
#     files=np.array(files)

#     '''
#     get info about files
#     '''

#     model_name=[]
#     for file in files:
#         model_name.append(file.split('_')[5])

#     model_name = np.array(model_name)
#     model_name_unique = np.unique(model_name)
#     print 'models: '+ np.str(model_name_unique)

#     file_year=[]
#     for file in files:
#         tmp = file.split('_')[8]
#         file_year.append(np.int(tmp[7:11]))

#     file_year = np.array(file_year)


#     '''
#     read in cmip5 tas
#     '''

#     all_cubes=[]
#     model_names_final = []
#     for i,model in enumerate(model_name_unique):
#         print model
#         loc = np.where((model_name == model) & (file_year <= 2100) )
#         files_tmp=files[loc[0]]
#         if np.size(files_tmp) >= 1:
#             while True:
#                 try:
#                     cube = iris.load_cube(files_tmp,'air_temperature',callback=my_callback)
#                     all_cubes.append(monthly_to_yearly(cube))
#                     model_names_final.append(model)
#                     break
#                 except iris.exceptions.ConstraintMismatchError:
#                     cubes = iris.load_raw(files_tmp,'air_temperature',callback=my_callback)
#                     cube = iris.experimental.concatenate.concatenate(cubes)
#                     all_cubes.append(monthly_to_yearly(cube[0]))
#                     model_names_final.append(model)
#                     print 'had to read in the hard way...'
#                     break


#     '''
#     and calculate timeseries
#     '''

#     all_cubes_timeseries = []
#     all_cubes_timeseries_year = []
#     for i,cube in enumerate(all_cubes):
#         print i
#         if not cube.coord('latitude').has_bounds():
#             cube.coord('latitude').guess_bounds()
#         if not cube.coord('longitude').has_bounds():
#             cube.coord('longitude').guess_bounds()
#         grid_areas = iris.analysis.cartography.area_weights(cube)
#         all_cubes_timeseries.append(cube.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
#         coord = cube.coord('time')
#         year = np.array([coord.units.num2date(value).year for value in coord.points])
#         all_cubes_timeseries_year.append(year)

#     model_names_final_all_models.append(model_names_final)
#     all_cubes_all_models.append(all_cubes)
#     all_cubes_timeseries_all_models.append(all_cubes_timeseries)
#     all_cubes_timeseries_year_all_models.append(all_cubes_timeseries_year)

# '''
# plotting...
# '''


# '''
# all
# '''

# colours = ['black','blue','green','yellow','red']
# lns = []
# for i,all_cubes_timeseries in enumerate(all_cubes_timeseries_all_models):
#     for j,cube in enumerate(all_cubes_timeseries):
#         model = model_names_final_all_models[i][j]
#         hist_model_loc = np.where(model_names_final_all_models[0] == np.array(model))
#         if np.size(hist_model_loc) >= 1:
#             all_cubes_timeseries_hist = all_cubes_timeseries_all_models[0][hist_model_loc[0]]
#             all_cubes_timeseries_year = all_cubes_timeseries_year_all_models[i]
#             ln = plt.plot(all_cubes_timeseries_year[j],all_cubes_timeseries[j].data - np.mean(all_cubes_timeseries_hist.data[-45:-5]),colours[i],label = run_names[i])
#     lns += ln

# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs,loc = 2).draw_frame(False)
# plt.xlabel('year')
# plt.ylabel('temperature anomaly from years 1960-2000 (K)')
# mpl.rcParams['xtick.major.pad'] = 10
# mpl.rcParams['ytick.major.pad'] = 10

# plt.show()


# '''
# rcps
# '''

# for i,dummy in enumerate(run_names):

#     lns = []

#     all_cubes_timeseries = all_cubes_timeseries_all_models[i]
#     for j,cube in enumerate(all_cubes_timeseries):
#         model = model_names_final_all_models[i][j]
#         hist_model_loc = np.where(model_names_final_all_models[0] == np.array(model))
#         if np.size(hist_model_loc) >= 1:
#             all_cubes_timeseries_hist = all_cubes_timeseries_all_models[0][hist_model_loc[0]]
#             all_cubes_timeseries_year = all_cubes_timeseries_year_all_models[i]
#             ln = plt.plot(all_cubes_timeseries_year[j][1::],all_cubes_timeseries[j].data[1::] - np.mean(all_cubes_timeseries_hist.data[-45:-5]),label = model_names_final_all_models[i][j])
#             lns += ln

#     labs = [l.get_label() for l in lns]
#     plt.legend(lns, labs,loc = 2,prop={'size':7}).draw_frame(False)
#     plt.title(run_names[i])
#     plt.xlabel('year')
#     plt.ylabel('temperature anomaly from years 1960-2000 (K)')
#     mpl.rcParams['xtick.major.pad'] = 10
#     mpl.rcParams['ytick.major.pad'] = 10

#     plt.show()


# '''
# max
# '''

# max_model = []
# models = []

# #rcp85
# i = 4

# all_cubes_timeseries = all_cubes_timeseries_all_models[i]
# for j,cube in enumerate(all_cubes_timeseries):
#     model = model_names_final_all_models[i][j]
#     hist_model_loc = np.where(model_names_final_all_models[0] == np.array(model))
#     if np.size(hist_model_loc) >= 1:
#         all_cubes_timeseries_hist = all_cubes_timeseries_all_models[0][hist_model_loc[0]]
#         all_cubes_timeseries_year = all_cubes_timeseries_year_all_models[i]
#         test = all_cubes_timeseries[j].data[1::]
#         if np.size(test) >= 1: 
#             x = np.max(all_cubes_timeseries[j].data[1::] - np.mean(all_cubes_timeseries_hist.data[-45:-5]))
#             max_model.append(x)
#             models.append(model)
#         if np.size(test) == 0:
#             max_model.append(np.nan)
#             models.append(model)

# max_model = np.array(max_model)
# loc = np.where(max_model == np.max(max_model[np.logical_not(np.isnan(max_model))]))
# models = np.array(models)
# highest_model = models[loc[0]]


# #rcp85
# tmp = np.array(model_names_final_all_models[i])
# loc = np.where(tmp == highest_model)

# #hist
# tmp = np.array(model_names_final_all_models[0])
# loc2 = np.where(tmp == highest_model)

# highest_model_results = all_cubes_all_models[i][loc[0]]
# highest_model_results_historical = all_cubes_all_models[0][loc2[0]]

# hist_mean = highest_model_results_historical[-45:-5].collapsed('time',iris.analysis.MEAN)
# rcp85_2090_2100_mean = highest_model_results[-11:-1].collapsed('time',iris.analysis.MEAN)

# diff = iris.analysis.maths.subtract(rcp85_2090_2100_mean,hist_mean)
# diff.standard_name = 'air_temperature_anomaly'

# plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
# qplt.contourf(diff,np.linspace(0,17,50))
# plt.gca().coastlines()
# plt.show()

# '''
# cycle through models for animation
# '''

# for model in np.array(model_names_final_all_models[i]):

#     #rcp85
#     tmp = np.array(model_names_final_all_models[i])
#     loc = np.where(tmp == model)

#     #hist
#     tmp = np.array(model_names_final_all_models[0])
#     loc2 = np.where(tmp == model)

#     highest_model_results = all_cubes_all_models[i][loc[0]]
#     highest_model_results_historical = all_cubes_all_models[0][loc2[0]]
    
#     while True:
#         try:
#             coord = highest_model_results.coord('time')
#             year = np.array([coord.units.num2date(value).year for value in coord.points])
#             loc = np.where(year >= 2099)

#             coord = highest_model_results_historical.coord('time')
#             year = np.array([coord.units.num2date(value).year for value in coord.points])
#             loc2 = np.where((year >= 1961) & (year <= 2000))

#             hist_mean = highest_model_results_historical[loc2].collapsed('time',iris.analysis.MEAN)
#             rcp85_2090_2100_mean = highest_model_results[loc].collapsed('time',iris.analysis.MEAN)

#             diff = iris.analysis.maths.subtract(rcp85_2090_2100_mean,hist_mean)
#             diff.standard_name = 'air_temperature_anomaly'

#             plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
#             qplt.contourf(diff,np.linspace(-5,25,50))
#             plt.gca().coastlines()
#             plt.savefig('/home/ph290/Documents/teaching/masters/rcp_85_2100_maps/'+model+'.png')
#             break
#         except (IndexError, ValueError):
#             print 'issue with: '+model
#             break


# '''
# historical v obs
# '''

# t_file = '/home/ph290/data1/observations/hadcrut4/HadCRUT.4.2.0.0.median.nc'

# t = iris.load_cube(t_file,'near_surface_temperature_anomaly')
# t_annual = monthly_to_yearly(t)
# t_annual.coord('latitude').guess_bounds()
# t_annual.coord('longitude').guess_bounds()
# grid_areas = iris.analysis.cartography.area_weights(t_annual)
# global_avg = t_annual.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=grid_areas)
# coord = global_avg.coord('time')
# year = np.array([coord.units.num2date(value).year for value in coord.points])

# i = 0
# all_cubes_timeseries = all_cubes_timeseries_all_models[0]
# for j,cube in enumerate(all_cubes_timeseries):
#     model = model_names_final_all_models[i][j]
#     hist_model_loc = np.where(model_names_final_all_models[0] == np.array(model))
#     if np.size(hist_model_loc) >= 1:
#         all_cubes_timeseries_hist = all_cubes_timeseries_all_models[0][hist_model_loc[0]]
#         all_cubes_timeseries_year = all_cubes_timeseries_year_all_models[i]
#         year2 = all_cubes_timeseries_year[j]
#         loc = np.where((year2 >= 1961) & (year2 <= 2000))
#         ln1 = plt.plot(all_cubes_timeseries_year[j],all_cubes_timeseries[j].data - np.mean(all_cubes_timeseries_hist.data[loc]),'k',label = 'CMIP5 model')

# loc = np.where((year >= 1961) & (year <= 2000))
# ln2 = plt.plot(year,global_avg.data-np.mean(global_avg.data[loc]),'r',linewidth=2,label = 'HadCRUT4 observatoinal product')

# lns = ln1+ln2
# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs,loc = 2).draw_frame(False)
# plt.xlabel('year')
# plt.ylabel('temperature anomaly from years 1960-2000 (K)')
# plt.xlim(1860,2005)
# plt.ylim(-1.5,1.0)
# mpl.rcParams['xtick.major.pad'] = 10
# mpl.rcParams['ytick.major.pad'] = 10

# plt.savefig('/home/ph290/Desktop/cmip5_hadcrut4.ps')
# #plt.show()


# loc = np.where((year >= 1961) & (year <= 2000))
# t_annual_meaned = t_annual.collapsed('time',iris.analysis.MEAN)
# qplt.contourf(t_annual_meaned)
# plt.gca().coastlines()
# plt.show()

'''
'''

t_file = '/home/ph290/data1/observations/hadcrut4/HadCRUT.4.2.0.0.median.nc'

t = iris.load_cube(t_file,'near_surface_temperature_anomaly')
t_annual = monthly_to_yearly(t)
t_annual.coord('latitude').guess_bounds()
t_annual.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(t_annual)
global_avg = t_annual.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=grid_areas)
coord = global_avg.coord('time')
year_obs = np.array([coord.units.num2date(value).year for value in coord.points])

colours = ['black','blue','green','yellow','red']
lns = []
for i,all_cubes_timeseries in enumerate(all_cubes_timeseries_all_models):
    for j,cube in enumerate(all_cubes_timeseries):
        model = model_names_final_all_models[i][j]
        hist_model_loc = np.where(model_names_final_all_models[0] == np.array(model))
        if np.size(hist_model_loc) >= 1:
            all_cubes_timeseries_hist = all_cubes_timeseries_all_models[0][hist_model_loc[0]]
            all_cubes_timeseries_year = all_cubes_timeseries_year_all_models[i]
            ln = plt.plot(all_cubes_timeseries_year[j],all_cubes_timeseries[j].data - np.mean(all_cubes_timeseries_hist.data[-45:-5]),colours[i],label = run_names[i])
    lns += ln

years = all_cubes_timeseries_year_all_models[0][0]

mean_data=[]

for yr in years:
    tmp = []
    for i,all_cubes_timeseries in enumerate(all_cubes_timeseries_all_models[0]):
        loc = np.where(all_cubes_timeseries_year_all_models[0][i] == yr)
        if np.size(loc) >= 1:
            tmp = np.append(tmp,all_cubes_timeseries[loc[0]].data)
    mean_data = np.append(mean_data,np.mean(tmp[np.logical_not(np.isnan(tmp))]))

plt.plot(years,mean_data - np.mean(mean_data[-45:-5]),'r',linewidth = 2)

plt.plot(year_obs,global_avg.data,'g',linewidth = 2)


labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 2).draw_frame(False)
plt.xlabel('year')
plt.ylabel('temperature anomaly from years 1960-2000 (K)')
mpl.rcParams['xtick.major.pad'] = 10
mpl.rcParams['ytick.major.pad'] = 10

#plt.show()
plt.savefig('/home/ph290/Desktop/camilo.ps')
