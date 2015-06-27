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
import monthly_to_yearly
import scipy
from scipy import signal
from scipy.signal import butter, lfilter




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



directory = '/data/data0/ph290/cmip5_data/'
files1 = np.array(glob.glob(directory+'hadgem2es/sea_surface_height/*.nc'))
files2 = np.array(glob.glob(directory+'hadgem2_cc/sea_surface_height/*.nc'))

#files1 = np.array(glob.glob(directory+'hadgem2es/psl/*.nc'))
#files2 = np.array(glob.glob(directory+'hadgem2_cc/psl/*.nc'))

es_cubes = iris.load(files1,'sea_surface_height_above_geoid',callback = my_callback)
cc_cubes = iris.load(files2,'sea_surface_height_above_geoid',callback = my_callback)
#es_cubes = iris.load(files1,'air_pressure_at_sea_level',callback = my_callback)
#cc_cubes = iris.load(files2,'air_pressure_at_sea_level',callback = my_callback)

es_cube = iris.experimental.concatenate.concatenate(es_cubes)
cc_cube = iris.experimental.concatenate.concatenate(cc_cubes)

es_cube_ann = monthly_to_yearly.monthly_to_yearly(es_cube[0])
cc_cube_ann = monthly_to_yearly.monthly_to_yearly(cc_cube[0])
es_cube_ann.data = scipy.signal.detrend(es_cube_ann.data, axis=0)
cc_cube_ann.data = scipy.signal.detrend(cc_cube_ann.data, axis=0)

coord = cc_cube_ann.coord('time')
cc_year = np.array([coord.units.num2date(value).year for value in coord.points])
coord = es_cube_ann.coord('time')
es_year = np.array([coord.units.num2date(value).year for value in coord.points])

tsi_file = '/home/ph290/data0/cmip5_data/hadgem2_forcing_data/scvary_l09a.dat'
tsi = np.genfromtxt(tsi_file)

# plt.plot(tsi[:,0],tsi[:,1])




def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

fs = 10000.0
lowcut = 500.0
highcut = 1.0e50

tsi_filtered = butter_bandpass_filter(tsi[:,1], lowcut, highcut, fs, order=6)

tsi_years = tsi[159::,0]
tsi_data = tsi_filtered[159::]

plt.plot(tsi[:,0],tsi[:,1])

high_years = np.where(tsi_data > 0.2)
low_years = np.where(tsi_data < 0.2)

high_years = np.where(tsi_data[0:cc_year.size] > 0.3)
low_years = np.where(tsi_data[0:cc_year.size] < -0.3)
cc_high_avg = cc_cube_ann[high_years].collapsed('time',iris.analysis.MEAN)
cc_low_avg = cc_cube_ann[low_years].collapsed('time',iris.analysis.MEAN)


high_years = np.where(tsi_data[0:es_year.size] > 0.3)
low_years = np.where(tsi_data[0:es_year.size] < -0.3)
es_high_avg = es_cube_ann[high_years].collapsed('time',iris.analysis.MEAN)
es_low_avg = es_cube_ann[low_years].collapsed('time',iris.analysis.MEAN)


plt.figure()
qplt.contourf(iris.analysis.maths.subtract(es_high_avg,es_low_avg),30)
plt.gca().coastlines()
plt.title('no stratosphere')
plt.show()

plt.figure()
qplt.contourf(iris.analysis.maths.subtract(cc_high_avg,cc_low_avg),30)
plt.gca().coastlines()
plt.title('stratosphere')
plt.show()
