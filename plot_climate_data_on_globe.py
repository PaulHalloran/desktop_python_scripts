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

# run_names = ['rcp85']

# model_names_final_all_models = []
# all_cubes_all_models = []
# all_cubes_timeseries_all_models = []
# all_cubes_timeseries_year_all_models = []

# run_name = run_names[0]


# print 'processing '+run_name

# files=glob.glob(directory+run_name+'/tas_Amon*.nc')
# files=np.array(files)

# '''
# get info about files
# '''

# model_name=[]
# for file in files:
#     model_name.append(file.split('_')[5])

# model_name = np.array(model_name)
# model_name_unique = np.unique(model_name)
# print 'models: '+ np.str(model_name_unique)

# file_year=[]
# for file in files:
#     tmp = file.split('_')[8]
#     file_year.append(np.int(tmp[7:11]))

# file_year = np.array(file_year)


# '''
# read in cmip5 tas
# '''

# all_cubes=[]
# model_names_final = []
# for i,model in enumerate(model_name_unique):
#     print model
#     loc = np.where((model_name == model) & (file_year <= 2100) )
#     files_tmp=files[loc[0]]
#     if np.size(files_tmp) >= 1:
#         while True:
#             try:
#                 cube = iris.load_cube(files_tmp,'air_temperature',callback=my_callback)
#                 all_cubes.append(monthly_to_yearly(cube))
#                 model_names_final.append(model)
#                 break
#             except iris.exceptions.ConstraintMismatchError:
#                 cubes = iris.load_raw(files_tmp,'air_temperature',callback=my_callback)
#                 cube = iris.experimental.concatenate.concatenate(cubes)
#                 all_cubes.append(monthly_to_yearly(cube[0]))
#                 model_names_final.append(model)
#                 print 'had to read in the hard way...'
#                 break

# ann_data =  all_cubes[31]


# x = [0,19,39,59,69,79,89]

# for tmp,i in enumerate(x):

#     def main():
#         fig = plt.figure()
#         ax = plt.axes(projection=ccrs.Orthographic())

#         # make the map global rather than have it zoom in to
#         # the extents of any plotted data
#         ax.set_global()

#         #ax.stock_img()

#         #plt.plot(-0.08, 51.53, 'o', transform=ccrs.PlateCarree())
#         qplt.contourf(ann_data[i],np.linspace(220,320,21))
#     #[210,220,230,240,250,260,270,280,290,300,310,320,330])
#         #plt.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.PlateCarree())
#         #plt.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.Geodetic())
#         ax.coastlines()
#         plt.show()
#         #plt.savefig('/home/ph290/Documents/teaching/masters/mpi_rcp95_globes/fig_'+np.str(tmp)+'.ps')

    # if __name__ == '__main__':
    #     main()



x = 10
fig = plt.figure()
ax = plt.axes(projection=ccrs.Orthographic())
# make the map global rather than have it zoom in to
# the extents of any plotted data
ax.set_global()
#ax.stock_img()
#plt.plot(-0.08, 51.53, 'o', transform=ccrs.PlateCarree())
qplt.contourf(ann_data[i],np.linspace(250,310,21))
#[210,220,230,240,250,260,270,280,290,300,310,320,330])
#plt.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.PlateCarree())
#plt.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.Geodetic())
ax.coastlines()
#plt.show()
plt.savefig('/home/ph290/Documents/teaching/masters/mpi_rcp95_globes/fig.pdf')


