
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
import iris.plot as iplt

def digital(cube):
    cube_out = cube
    cube_out.data[np.where(cube.data >= 0.0)] = 1.0
    cube_out.data[np.where(cube.data < 0.0)] = -1.0
    return cube_out

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
files1 = '/home/ph290/data0/cmip5_data/sstClim/tas/'
files2 = '/home/ph290/data0/cmip5_data/sstClimAerosol/tas/'
files3 = '/home/ph290/data0/cmip5_data/sstClim/pr/'
files4 = '/home/ph290/data0/cmip5_data/sstClimAerosol/pr/'

'''

'''

models = ['HadGEM2-A','MIROC5','NorESM1-M']


'''
Temperature
'''

files = np.array(glob.glob(files1+'*'+models[0]+'*.nc'))
HadGEM2_sstClim_tas = iris.load(files,'air_temperature',callback = my_callback)
HadGEM2_sstClim_tas = HadGEM2_sstClim_tas[0].collapsed('time',iris.analysis.MEAN)
files = np.array(glob.glob(files2+'*'+models[0]+'*.nc'))
HadGEM2_sstClimAero_tas = iris.load(files,'air_temperature',callback = my_callback)
HadGEM2_sstClimAero_tas = HadGEM2_sstClimAero_tas[0].collapsed('time',iris.analysis.MEAN)

files = np.array(glob.glob(files1+'*'+models[1]+'*.nc'))
MIROC5_sstClim_tas = iris.load(files,'air_temperature',callback = my_callback)
MIROC5_sstClim_tas = MIROC5_sstClim_tas[0].collapsed('time',iris.analysis.MEAN)
files = np.array(glob.glob(files2+'*'+models[1]+'*.nc'))
MIROC5_sstClimAero_tas = iris.load(files,'air_temperature',callback = my_callback)
MIROC5_sstClimAero_tas = MIROC5_sstClimAero_tas[0].collapsed('time',iris.analysis.MEAN)

#note, this is currently unavaliable on CMIP5
#files = np.array(glob.glob(files1+'*'+models[2]+'*.nc'))
#NorESM1_sstClim_tas = iris.load(files,'air_temperature',callback = my_callback)
#files = np.array(glob.glob(files2+'*'+models[2]+'*.nc'))
#NorESM1_sstClimAero_tas = iris.load(files,'air_temperature',callback = my_callback)

#dummy cube
latitude = DimCoord(range(-90, 90, 5), standard_name='latitude',
                    units='degrees')
longitude = DimCoord(range(0, 360, 5), standard_name='longitude',
                     units='degrees')
cube = iris.cube.Cube(np.zeros((18*2, 36*2), np.float32),
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)])


'''
Precip
'''

files = np.array(glob.glob(files3+'*'+models[0]+'*.nc'))
HadGEM2_sstClim_pr = iris.load(files,'precipitation_flux',callback = my_callback)
HadGEM2_sstClim_pr = HadGEM2_sstClim_pr[0].collapsed('time',iris.analysis.MEAN)
files = np.array(glob.glob(files4+'*'+models[0]+'*.nc'))
HadGEM2_sstClimAero_pr = iris.load(files,'precipitation_flux',callback = my_callback)
HadGEM2_sstClimAero_pr = HadGEM2_sstClimAero_pr[0].collapsed('time',iris.analysis.MEAN)

files = np.array(glob.glob(files3+'*'+models[1]+'*.nc'))
MIROC5_sstClim_pr = iris.load(files,'precipitation_flux',callback = my_callback)
MIROC5_sstClim_pr = MIROC5_sstClim_pr[0].collapsed('time',iris.analysis.MEAN)
files = np.array(glob.glob(files4+'*'+models[1]+'*.nc'))
MIROC5_sstClimAero_pr = iris.load(files,'precipitation_flux',callback = my_callback)
MIROC5_sstClimAero_pr = MIROC5_sstClimAero_pr[0].collapsed('time',iris.analysis.MEAN)

files = np.array(glob.glob(files3+'*'+models[2]+'*.nc'))
NorESM1_sstClim_pr = iris.load(files,'precipitation_flux',callback = my_callback)
NorESM1_sstClim_pr = NorESM1_sstClim_pr[0].collapsed('time',iris.analysis.MEAN)
files = np.array(glob.glob(files4+'*'+models[2]+'*.nc'))
NorESM1_sstClimAero_pr = iris.load(files,'precipitation_flux',callback = my_callback)
NorESM1_sstClimAero_pr = NorESM1_sstClimAero_pr[0].collapsed('time',iris.analysis.MEAN)

HadGEM2_tas = iris.analysis.maths.subtract(HadGEM2_sstClimAero_tas,HadGEM2_sstClim_tas)
MIROC5_tas = iris.analysis.maths.subtract(MIROC5_sstClimAero_tas,MIROC5_sstClim_tas)

HadGEM2_pr = iris.analysis.maths.subtract(HadGEM2_sstClimAero_pr,HadGEM2_sstClim_pr)
MIROC5_pr = iris.analysis.maths.subtract(MIROC5_sstClimAero_pr,MIROC5_sstClim_pr)
NorESM1_pr = iris.analysis.maths.subtract(NorESM1_sstClimAero_pr,NorESM1_sstClim_pr)

destination_cube = NorESM1_pr.copy()
HadGEM2_tas_regrid  = iris.analysis.interpolate.regrid(HadGEM2_tas, destination_cube)
MIROC5_tas_regrid  = iris.analysis.interpolate.regrid(MIROC5_tas, destination_cube)
HadGEM2_pr_regrid  = iris.analysis.interpolate.regrid(HadGEM2_pr, destination_cube)
MIROC5_pr_regrid  = iris.analysis.interpolate.regrid(MIROC5_pr, destination_cube)

tas_mean = iris.analysis.maths.add(HadGEM2_tas_regrid,MIROC5_tas_regrid)
tas_mean2 = iris.analysis.maths.divide(tas_mean,2)

HadGEM2_tas_regrid_digital = digital(iris.analysis.interpolate.regrid(HadGEM2_tas_regrid,cube)).copy()
MIROC5_tas_regrid_digital = digital(iris.analysis.interpolate.regrid(MIROC5_tas_regrid,cube)).copy()
tas_regrid_digital = iris.analysis.maths.add(HadGEM2_tas_regrid_digital,MIROC5_tas_regrid_digital)
tas_regrid_digital2 = tas_regrid_digital.copy()
tas_regrid_digital2.data = tas_regrid_digital2.data*0.0+np.nan
tas_regrid_digital2.data[np.where(tas_regrid_digital.data == 2)] = 1.0
tas_regrid_digital2.data[np.where(tas_regrid_digital.data == -2)] = 1.0

# plt.figure()
# qplt.contourf(tas_mean2,50)
# points = iplt.points(tas_regrid_digital2, c = tas_regrid_digital2.data, s= 2.0)
# plt.gca().coastlines()
# plt.title('Aerosol mean surface temperature change and agreement (K)')
# plt.show()

north = 85
south = -20
east = 40
west = -130

tmp = tas_mean2.intersection(longitude=(west, east))
tmp = tmp.intersection(latitude=(south, north))

tmp2 = tas_regrid_digital2.intersection(longitude=(west, east))
tmp2 = tmp2.intersection(latitude=(south, north))

plt.close('all')
plt.figure()
minz = -0.3
maxz = 0.3
tmpb = tmp.copy()
tmpb.data[np.where(tmpb.data < minz)] = minz
tmpb.data[np.where(tmpb.data > maxz)] = maxz
qplt.contourf(tmpb,np.linspace(minz,maxz,51))
points = iplt.points(tmp2, c = tmp2.data , s= 2.0)
plt.gca().coastlines()
plt.title('Aerosol mean surface temperature change and agreement (K)')
plt.savefig('/home/ph290/Documents/figures/tas_aero.png')
plt.show(block = False)


NorESM1_pr_regrid = NorESM1_pr.copy()
pr_mean = iris.analysis.maths.add(HadGEM2_pr_regrid,MIROC5_pr_regrid)
#pr_mean = iris.analysis.maths.add(pr_mean,NorESM1_pr_regrid)
pr_mean2 = iris.analysis.maths.divide(pr_mean,2)

HadGEM2_pr_regrid_digital = digital(iris.analysis.interpolate.regrid(HadGEM2_pr_regrid,cube)).copy()
MIROC5_pr_regrid_digital = digital(iris.analysis.interpolate.regrid(MIROC5_pr_regrid,cube)).copy()
#NorESM1_pr_regrid_digital = digital(iris.analysis.interpolate.regrid(NorESM1_pr_regrid,cube))
pr_regrid_digital = iris.analysis.maths.add(HadGEM2_pr_regrid_digital,MIROC5_pr_regrid_digital)
#pr_regrid_digital = iris.analysis.maths.add(pr_regrid_digital,NorESM1_pr_regrid_digital)
pr_regrid_digital2 = pr_regrid_digital.copy()
pr_regrid_digital2.data = pr_regrid_digital2.data*0.0+np.nan
pr_regrid_digital2.data[np.where(pr_regrid_digital.data == 2)] = 1.0
pr_regrid_digital2.data[np.where(pr_regrid_digital.data == -2)] = 1.0

plt.figure()
qplt.contourf(pr_mean2,np.linspace(-0.0000075,0.0000075,50))
points = iplt.points(pr_regrid_digital2, c = pr_regrid_digital2.data, s= 2.0)
plt.gca().coastlines()
plt.title('Aerosol mean precipitation change and agreement')
plt.show()

