import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris
import glob
import iris.experimental.concatenate
import iris.analysis
import iris.quickplot as qplt
import iris.analysis.cartography
import cartopy.crs as ccrs
import subprocess
from iris.coords import DimCoord
import iris.coord_categorisation
import matplotlib as mpl
import gc


def my_callback(cube,field, files):
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('history')
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    cube.attributes.pop('table_id')
    #if np.size(cube) > 1:
    #cube = iris.experimental.concatenate.concatenate(cube)
    return cube

def extract_data(data_in):
    data_out = data_in.data
    return data_out
    
def regrid_data_0(file,variable_name,out_filename):
    p = subprocess.Popen("cdo remapbil,r360x180 -selname,"+variable_name+" "+file+" "+out_filename,shell=True)
    p.wait()

directory = '/home/ph290/data0/cmip5_data/'

pr_files = np.array(glob.glob(directory+'pr/*.nc'))

'''
which models do we have?
'''

models = []
for file in pr_files:
    models.append(file.split('/')[-1].split('_')[2])

models_unique = np.unique(np.array(models))


'''
reads in and regrid data into a single 360x180 file per model - note, this only needs to be done once
'''

tmp_dir = '/data/data0/ph290/cmip5_data/tmp_dir/'

final_models1 = []
cubes = []
for i,model in enumerate(models_unique):
	# print i
	model = models_unique[0]
	file = glob.glob(directory+'pr/'+'*'+model+'*.nc')
	cube = iris.load(file,'rainfall_flux')
	cube = iris.experimental.concatenate.concatenate(cube)
	cube = cube[0]
	tmp_cube = cube[0].copy()
	iris.coord_categorisation.add_year(cube, 'time', name='year2')
	my_years_constraint = iris.Constraint(year2=lambda y: y in range(1965,1984))
	cube_extract= cube.extract(my_years_constraint)
	mean_cube = cube_extract.collapsed('time',iris.analysis.MEAN)
	tmp_cube.data = mean_cube.data
	iris.fileformats.netcdf.save(tmp_cube, tmp_dir+'delete.nc', netcdf_format='NETCDF3_CLASSIC')
	out_filename = directory+'pr/regridded/'+model+'.nc'
	regrid_data_0(tmp_dir+'delete.nc','rainfall_flux',out_filename)
	#note that the regridding step messus up the depth coordinate...
	final_models1.append(model)
	print model + ' done'

	
#  Underlying netCDF file format, one of 'NETCDF4', 'NETCDF4_CLASSIC',
#         'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. Default is 'NETCDF4' format.

# '''
# Read in 1x1 degree basin masks from WOA
# '''
# 
# woa_dir = '/home/ph290/Documents/teaching/'
# basin_mask = iris.load_cube(woa_dir+'basin.nc')
# basin_mask_tmp = basin_mask[0][0]
# basin_mask_tmp.data[np.where(np.logical_not(basin_mask_tmp.data == 1))] = np.nan
# # 1 = Atlantic
# # 2 = Pacific
# # 3 = Indian
# # 4 = Southen Ocean
# # 5 = Arctic
# # if upside down:
# # basin_mask_flipped = iris.analysis.maths.np.flipud(basin_mask.data)
# # apply mask with:
# # sstb = iris.analysis.maths.multiply(sst,basin_mask_flipped)
# 
# '''
# Read in GLODAP and WOA data
# '''
# 
# qlodap_dir = '/home/ph290/data1/observations/glodap/'
# woa_dir = '/home/ph290/Documents/teaching/'
# alk_in = iris.load_cube(qlodap_dir+'PALK.nc','Potential_Alkalinity')
# alk_in = iris.analysis.maths.divide(alk_in,1000.0) # convert units
# alk_in.transpose((0,2,1)) #reorders dimenions to be same as CMIP5 and WOA
# salinity_in = iris.load_cube(woa_dir+'salinity_annual_1deg.nc','sea_water_salinity')
# 
# 
# 
# '''
# Reading file details (model name etc)
# '''
# 
# alk_files_regrid = np.array(glob.glob(directory+'talk/regridded/*.nc'))
# 
# model_names = []
# for file in alk_files_regrid:
#     tmp = file.split('/')[-1].split('_')[0].split('.')[0]
#     model_names.append(tmp)
#   
# '''
# Reading in regridded files
# '''
# 
# model_names = np.array(model_names)
# model_names_unique = np.unique(model_names)
# 
# alk_cubes = []
# for model in model_names_unique:
#     print model
#     files = np.array(glob.glob(directory+'talk/regridded/*'+model+'*.nc'))
#     cube = iris.load(files)
#     alk_cubes.append(cube[0])
# 
# so_cubes = []
# for model in model_names_unique:
#     print model
#     files = np.array(glob.glob(directory+'so/regridded/*'+model+'*.nc'))
#     cube = iris.load(files)
#     cube = cube[0]
#     mean_tmp = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN)
#     if mean_tmp[0].data <= 1:
#         cube = iris.analysis.maths.multiply(cube,1000.0)
#         print 'converting model because in incorrect units'
#     so_cubes.append(cube)
# 
# basins = np.array([0,1,2,3,10,11])
# basin_names = np.array(['Global','Atlantic','Pacific','Indian','Southern Ocean','Arctic Ocean'])
# 
# '''
# Depths
# '''
# 
# depths = []
# for model in model_names_unique:
#     print model
#     count = 0
#     files = glob.glob(directory+'talk/'+'*'+model+'*.nc')
#     cube = iris.load(files[0],'sea_water_alkalinity_expressed_as_mole_equivalent')
#     try:
#         depth = cube[0].coord('ocean depth coordinate').points
#     except iris.exceptions.CoordinateNotFoundError:
#         print 'depth name not ocean depth coordinate'
#         count += 1
#     try:
#         depth = cube[0].coord('depth').points
#     except iris.exceptions.CoordinateNotFoundError:
#         print 'depth name not depth'
#         count += 1
#     try:
#         depth = cube[0].coord('ocean sigma over z coordinate').points
#     except iris.exceptions.CoordinateNotFoundError:
#         print 'depth name not sigma'
#         count += 1
#     if count == 3:
#         print 'no depth coordinate was found'
#         break
#     depths.append(depth)
# 
# 
# 
# '''
# Calculate CMIP5 profiles
# '''
# 
# regional_profiles_alk = []
# regions = ['']
# 
# for i in basins:
#     model_profiles = []
#     for cube in alk_cubes:
#         if not cube.coord('latitude').has_bounds():
#             cube.coord('latitude').guess_bounds()
#         if not cube.coord('longitude').has_bounds():
#             cube.coord('longitude').guess_bounds()
#         grid_areas = iris.analysis.cartography.area_weights(cube)
#         basin_mask_tmp = basin_mask[0][0]
#         if i == 0:
#             #global case
#             loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
#             loc2 = np.where(basin_mask_tmp.data >= 1)
#         else:
#             #all other cases
#             loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
#             loc2 = np.where(basin_mask_tmp.data == i)
#         basin_mask_tmp.data[loc1] = 0.0
#         basin_mask_tmp.data[loc2] = 1.0
#         basin_mask_flipped = basin_mask_tmp
#         basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
#         cube_tmp = cube
#         cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
#         cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
#         data_shape = np.shape(cube_tmp.data)
#         a = ma.getmaskarray(alk_in[0].data)
#         a =np.roll(a,180,axis=1)
#         a2 = np.tile(a,(data_shape[0],1,1))
#         b = ma.getmaskarray(cube_tmp.data)
#         mask = a2 | b
#         #by combining the masks above, the CMIP5 data has exactky the same land sea mask as the WOA data
#         cube_tmp.data = ma.masked_array(cube_tmp.data,mask)
#         #qplt.contourf(cube_tmp[0])
#         #plt.gca().coastlines()
#         #plt.show()
#         model_profiles.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
#     regional_profiles_alk.append(model_profiles)
# 
# 
# '''
# and so
# '''
# 
# regional_profiles_so = []
# regions = ['']
# 
# for i in basins:
#     model_profiles = []
#     for cube in so_cubes:
#         if not cube.coord('latitude').has_bounds():
#             cube.coord('latitude').guess_bounds()
#         if not cube.coord('longitude').has_bounds():
#             cube.coord('longitude').guess_bounds()
#         grid_areas = iris.analysis.cartography.area_weights(cube)
#         basin_mask_tmp = basin_mask[0][0]
#         if i == 0:
#             #global case
#             loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
#             loc2 = np.where(basin_mask_tmp.data >= 1)
#         else:
#             #all other cases
#             loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
#             loc2 = np.where(basin_mask_tmp.data == i)
#         basin_mask_tmp.data[loc1] = 0.0
#         basin_mask_tmp.data[loc2] = 1.0
#         basin_mask_flipped = basin_mask_tmp
#         basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
#         cube_tmp = cube
#         cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
#         cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
#         data_shape = np.shape(cube_tmp.data)
#         a = ma.getmaskarray(alk_in[0].data)
#         a =np.roll(a,180,axis=1)
#         a2 = np.tile(a,(data_shape[0],1,1))
#         b = ma.getmaskarray(cube_tmp.data)
#         mask = a2 | b
#         #by combining the masks above, the CMIP5 data has exactky the same land sea mask as the WOA data
#         cube_tmp.data = ma.masked_array(cube_tmp.data,mask)
#         #qplt.contourf(cube_tmp[0])
#         #plt.gca().coastlines()
#         #plt.show()
#         model_profiles.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
#     regional_profiles_so.append(model_profiles)
# 
# 
# '''
# and alk/so (normalised)
# '''
# 
# cubes = np.copy(alk_cubes)
# for i in np.arange(len(cubes)):
#     cubes[i] = iris.analysis.maths.divide(alk_cubes[i],so_cubes[i])
# 
# regional_profiles_normal = []
# regions = ['']
# 
# for i in basins:
#     model_profiles = []
#     for cube in cubes:
#         if not cube.coord('latitude').has_bounds():
#             cube.coord('latitude').guess_bounds()
#         if not cube.coord('longitude').has_bounds():
#             cube.coord('longitude').guess_bounds()
#         grid_areas = iris.analysis.cartography.area_weights(cube)
#         basin_mask_tmp = basin_mask[0][0]
#         if i == 0:
#             #global case
#             loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
#             loc2 = np.where(basin_mask_tmp.data >= 1)
#         else:
#             #all other cases
#             loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
#             loc2 = np.where(basin_mask_tmp.data == i)
#         basin_mask_tmp.data[loc1] = 0.0
#         basin_mask_tmp.data[loc2] = 1.0
#         basin_mask_flipped = basin_mask_tmp
#         basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
#         cube_tmp = cube
#         cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
#         cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
#         data_shape = np.shape(cube_tmp.data)
#         a = ma.getmaskarray(alk_in[0].data)
#         a =np.roll(a,180,axis=1)
#         a2 = np.tile(a,(data_shape[0],1,1))
#         b = ma.getmaskarray(cube_tmp.data)
#         mask = a2 | b
#         #by combining the masks above, the CMIP5 data has exactky the same land sea mask as the WOA data
#         cube_tmp.data = ma.masked_array(cube_tmp.data,mask)
#         #qplt.contourf(cube_tmp[0])
#         #plt.gca().coastlines()
#         #plt.show()
#         model_profiles.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
#     regional_profiles_normal.append(model_profiles)
# 
# 
# '''
# Calculate GLODAP/WOA profiles
# '''
# 
# #alk_in = iris.load_cube(qlodap_dir+'PALK.nc','Potential_Alkalinity')
# #salinity_in = iris.load_cube(woa_dir+'salinity_annual_1deg.nc','sea_water_salinity')
# 
# so_in = salinity_in[0]
# so_in_data = np.roll(so_in.data,180,axis = 2)
# cube_tmp = alk_in.copy()
# cube_tmp.data = so_in_data
# cube = iris.analysis.maths.divide(alk_in,cube_tmp)
# 
# obs_profiles = []
# for i in basins:
#     if not cube.coord('latitude').has_bounds():
#         cube.coord('latitude').guess_bounds()
#     if not cube.coord('longitude').has_bounds():
#         cube.coord('longitude').guess_bounds()
#     grid_areas = iris.analysis.cartography.area_weights(cube)
#     basin_mask_tmp = basin_mask[0][0]
#     if i == 0:
#         #global case
#         loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
#         loc2 = np.where(basin_mask_tmp.data >= 1)
#     else:
#         #all other cases
#         loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
#         loc2 = np.where(basin_mask_tmp.data == i)
#     basin_mask_tmp.data[loc1] = 0.0
#     basin_mask_tmp.data[loc2] = 1.0
#     basin_mask_flipped = basin_mask_tmp
#     basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
#     basin_mask_flipped.data = iris.analysis.maths.np.roll(basin_mask_tmp.data,180,axis=1)
#     cube_tmp = cube
#     cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
#     cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
#     #qplt.contourf(cube_tmp[0])
#     #plt.gca().coastlines()
#     #plt.show()
#     obs_profiles.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
# 
# '''
# obs profile alk
# '''
# 
# cube = alk_in
# 
# obs_profiles_alk = []
# for i in basins:
#     if not cube.coord('latitude').has_bounds():
#         cube.coord('latitude').guess_bounds()
#     if not cube.coord('longitude').has_bounds():
#         cube.coord('longitude').guess_bounds()
#     grid_areas = iris.analysis.cartography.area_weights(cube)
#     basin_mask_tmp = basin_mask[0][0]
#     if i == 0:
#         #global case
#         loc1 = np.where(np.logical_not(basin_mask_tmp.data >= 1))
#         loc2 = np.where(basin_mask_tmp.data >= 1)
#     else:
#         #all other cases
#         loc1 = np.where(np.logical_not(basin_mask_tmp.data == i))
#         loc2 = np.where(basin_mask_tmp.data == i)
#     basin_mask_tmp.data[loc1] = 0.0
#     basin_mask_tmp.data[loc2] = 1.0
#     basin_mask_flipped = basin_mask_tmp
#     basin_mask_flipped.data = iris.analysis.maths.np.flipud(basin_mask_tmp.data)
#     basin_mask_flipped.data = iris.analysis.maths.np.roll(basin_mask_tmp.data,180,axis=1)
#     cube_tmp = cube
#     cube_tmp = iris.analysis.maths.multiply(cube,basin_mask_flipped)
#     cube_tmp.data = ma.masked_equal(cube_tmp.data,0)
#     #qplt.contourf(cube_tmp[0])
#     #plt.gca().coastlines()
#     #plt.show()
#     obs_profiles_alk.append(cube_tmp.collapsed(['latitude','longitude'], iris.analysis.MEAN,weights = grid_areas))
# 
# 
# '''
# Plot
# '''
# 
# linestyles = ['-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':','-', '-.', '--', ':']
# 
# for i,profiles in enumerate(regional_profiles_normal):
#     plt.figure()
#     print i
#     for j,model_profile in enumerate(profiles):
#         line = plt.plot(model_profile.data,depths[j]*(-1.0))
#         plt.setp(line, linestyle=linestyles[j],linewidth = 2)
#         #plt.title(basin_names[i])
#     #qplt.plot(obs_profiles[i],'k',linewidth = 3)
#     #print 'obs next'
#     #print obs_profiles[i].data
#     plt.plot(obs_profiles[i].data,obs_profiles[i].coord('depth').points*(-1.0),'k',linewidth = 3)
#     plt.title(basin_names[i])
#     #plt.show()
#     #plt.savefig('/home/ph290/Desktop/delete/'+basin_names[i]+'.png')
# 
# for i,profiles in enumerate(regional_profiles_alk):
#     plt.figure()
#     print i
#     for j,model_profile in enumerate(profiles):
#         line = plt.plot(model_profile.data,depths[j]*(-1.0))
#         plt.setp(line, linestyle=linestyles[j],linewidth = 2)
#         #plt.title(basin_names[i])
#     #qplt.plot(obs_profiles[i],'k',linewidth = 3)
#     #print 'obs next'
#     #print obs_profiles[i].data
#     plt.plot(obs_profiles_alk[i].data,obs_profiles_alk[i].coord('depth').points*(-1.0),'k',linewidth = 3)
#     plt.title(basin_names[i])
#     #plt.show()
#     plt.savefig('/home/ph290/Desktop/delete/'+basin_names[i]+'.png')
# 
# for j,model_profile in enumerate(profiles):
#     line = plt.plot([0,1],[j+1,j+1])
#     plt.text(1.2, j+0.8, model_names_unique[j],fontsize=12)
#     plt.setp(line, linestyle=linestyles[j],linewidth = 2)
#     plt.xlim([0,2])
#     plt.ylim([0,18])
# 
# #plt.show()
# plt.savefig('/home/ph290/Desktop/delete/lege.png')
# 
# #now need to  do the salinity side of things...

