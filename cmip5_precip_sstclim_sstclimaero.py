
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
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
import monthly_to_yearly as m2yr
from matplotlib import mlab
import matplotlib.mlab as ml
import cartopy
import cube_extract_region

def cube_extract_region(cube,min_lat,min_lon,max_lat,max_lon):
    region = iris.Constraint(longitude=lambda v: min_lon <= v <= max_lon,latitude=lambda v: min_lat <= v <= max_lat)
    return cube.extract(region)



def my_callback(cube,field, files):
    # there are some bad attributes in the NetCDF files which make the data incompatible for merging.
    cube.attributes.pop('tracking_id')
    cube.attributes.pop('creation_date')
    cube.attributes.pop('table_id')
    #if np.size(cube) > 1:
    #cube = iris.experimental.concatenate.concatenate(cube)


def extract_data(data_in):
    data_out = data_in.data
    return data_out


def regrid_data_0(file,variable_name,out_filename):
    p = subprocess.Popen("cdo remapbil,r360x180 -selname,"+variable_name+" "+file+" "+out_filename,shell=True)
    p.wait()
    return iris.load(out_filename)

directory = '/home/ph290/data0/cmip5_data/'
files1 = np.array(glob.glob('/home/ph290/data0/cmip5_data/pr/sstClim/*.nc'))
files2 = np.array(glob.glob('/home/ph290/data0/cmip5_data/pr/sstClimAerosol/*.nc'))

'''
which models do we have?
'''

models1 = []
for file in files1:
    models1.append(file.split('/')[-1].split('_')[2])


models_unique1 = np.unique(np.array(models1))

models2 = []
for file in files2:
    models2.append(file.split('/')[-1].split('_')[2])


models_unique2 = np.unique(np.array(models2))

models_unique = np.intersect1d(models_unique1,models_unique2)


cmip5_cube_mean_sstclim = []

for k,model in enumerate(models_unique):
    files = np.array(glob.glob('/home/ph290/data0/cmip5_data/pr/sstClim/*'+model+'*.nc'))
    cubes = iris.load(files,'rainfall_flux',callback = my_callback)
    if len(cubes) == 0:
        cubes = iris.load(files,'precipitation_flux',callback = my_callback)
    cube_tmp = cubes[0][0].copy()
    data = cube_tmp.data*0.0
    count = 0.0
    for j,cube in enumerate(cubes):
        print np.str(j)+' of '+np.str(len(cubes))
        cube = m2yr.monthly_to_yearly(cube)
        length = cube.shape[0]
        dim1_name = cube.coords()[0].long_name
        dim2_name = cube.coords()[1].long_name
        #for i,sub_cube in enumerate(cube.slices(['latitude', 'longitude'])):
        for i,sub_cube in enumerate(cube.slices([dim1_name,dim2_name])):
            print  np.str(i)+' of '+np.str(length)
            tmp_cube = sub_cube.copy()
            data = np.sum([data,tmp_cube.data],axis = 0, out=data)
            count += 1.0
    div_array = (cubes[0][0].copy().data*0.0)+count
    div_cube = cube_tmp.copy()
    div_cube.data = div_array
    data_cube = cube_tmp.copy()
    data_cube.data = data.copy()
    data_cube_mean = cube_tmp.copy()
    data_cube_mean = iris.analysis.maths.divide(data_cube,div_cube)
    cmip5_cube_mean_sstclim.append(data_cube_mean)
 

cmip5_cube_mean_sstclimaero = []

for k,model in enumerate(models_unique):
    files = np.array(glob.glob('/home/ph290/data0/cmip5_data/pr/sstClimAerosol/*'+model+'*.nc'))
    cubes = iris.load(files,'rainfall_flux',callback = my_callback)
    if len(cubes) == 0:
        cubes = iris.load(files,'precipitation_flux',callback = my_callback)
    cube_tmp = cubes[0][0].copy()
    data = cube_tmp.data*0.0
    count = 0.0
    for j,cube in enumerate(cubes):
        print np.str(j)+' of '+np.str(len(cubes))
        cube = m2yr.monthly_to_yearly(cube)
        length = cube.shape[0]
        dim1_name = cube.coords()[0].long_name
        dim2_name = cube.coords()[1].long_name
        #for i,sub_cube in enumerate(cube.slices(['latitude', 'longitude'])):
        for i,sub_cube in enumerate(cube.slices([dim1_name,dim2_name])):
            print  np.str(i)+' of '+np.str(length)
            tmp_cube = sub_cube.copy()
            data = np.sum([data,tmp_cube.data],axis = 0, out=data)
            count += 1.0
    div_array = (cubes[0][0].copy().data*0.0)+count
    div_cube = cube_tmp.copy()
    div_cube.data = div_array
    data_cube = cube_tmp.copy()
    data_cube.data = data.copy()
    data_cube_mean = cube_tmp.copy()
    data_cube_mean = iris.analysis.maths.divide(data_cube,div_cube)
    cmip5_cube_mean_sstclimaero.append(data_cube_mean)


'''
differences
'''

cmip5_cube_diff = []

for k,model in enumerate(models_unique):
    tmp1 = cmip5_cube_mean_sstclim[k]
    tmp2 = cmip5_cube_mean_sstclimaero[k]
    cmip5_cube_diff.append(iris.analysis.maths.subtract(tmp1,tmp2))


'''
plotting
'''

for k,model in enumerate(models_unique):
    plt.figure()
    qplt.contourf(cmip5_cube_diff[k],np.linspace(-0.00001,0.00001,50))
    plt.gca().coastlines()
    plt.title(model)
    plt.savefig('/home/ph290/Desktop/delete/'+model+'_pr_sstclim_sstclimaero.png')
    #plt.show()


indirect_effect_model_list = ['CSIRO-Mk3-6-0','HadGEM2-A','IPSL-CM5A-LR', 'MIROC5', 'MRI-CGCM3','NorESM1-M']
indirect_effect_models = []


for k,model in enumerate(models_unique):
    if np.intersect1d(model,indirect_effect_model_list):
        indirect_effect_models.append(cmip5_cube_diff[k])
        plt.figure()
        cube2 = cube_extract_region(cmip5_cube_diff[k],-20,180,80,360)
        qplt.contourf(cube2,np.linspace(-0.00001,0.00001,50))
        plt.gca().coastlines()
        plt.title(model)
        plt.savefig('/home/ph290/Desktop/delete/'+model+'_pr_sstclim_sstclimaero_indirect.png')
        #plt.show()
