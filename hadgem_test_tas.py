'''
import iris.plot as iplt
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
import running_mean
import matplotlib.cm as mpl_cm
from iris.coords import DimCoord
import iris.plot as iplt
import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
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
import iris
import iris.coord_categorisation
import iris.analysis

def m2a(cube):
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    return cube.aggregated_by('year', iris.analysis.MEAN)


dir1 = '/home/ph290/hadgem2/'

a  = iris.load('/home/ph290/hadgem2/ajnup/*.pp')[0]
b  = iris.load('/home/ph290/hadgem2/ajnuq/*.pp')[0]
c  = iris.load('/home/ph290/hadgem2/ajnuk/*.pp')[0]
d  = iris.load('/home/ph290/hadgem2/ajnuh/*.pp')[0]

tmp = np.empty([4 ,143, 145, 192])
tmp[0] = a.data[0:143,:,:]
tmp[1] = b.data[0:143,:,:]
tmp[2] = c.data[0:143,:,:]
tmp[3] = d.data[0:143,:,:]

ab = a.copy()
ab.data = np.mean(tmp,axis = 0)

coord = ab.coord('time')
dt = coord.units.num2date(coord.points)
year = np.array([coord.units.num2date(value).year for value in coord.points])

high_amo_loc = np.where((year >= 1930) & (year < 1950))[0]
low_amo_loc = np.where((year >= 1970) & (year < 1980))[0]

high_amo = ab[high_amo_loc].collapsed('time',iris.analysis.MEAN)
low_amo = ab[low_amo_loc].collapsed('time',iris.analysis.MEAN)

a1 = m2a(iris.load_cube('/home/ph290/data0/hadgem2es/tas_Amon_HadGEM2-ES_historical_r1i1p1.nc'))
b1 = m2a(iris.load_cube('/home/ph290/data0/hadgem2es/tas_Amon_HadGEM2-ES_historical_r2i1p1.nc'))
c1 = m2a(iris.load_cube('/home/ph290/data0/hadgem2es/tas_Amon_HadGEM2-ES_historical_r3i1p1.nc'))
d1 = m2a(iris.load_cube('/home/ph290/data0/hadgem2es/tas_Amon_HadGEM2-ES_historical_r4i1p1.nc'))
e1 = m2a(iris.load_cube('/home/ph290/data0/hadgem2es/tas_Amon_HadGEM2-ES_historical_r5i1p1.nc'))

tmp = np.empty([5 ,143, 145, 192])
tmp[0] = a1.data[0:143,:,:]
tmp[1] = b1.data[0:143,:,:]
tmp[2] = c1.data[0:143,:,:]
tmp[3] = d1.data[0:143,:,:]
tmp[4] = e1.data[0:143,:,:]

a1b = a.copy()
a1b.data = np.mean(tmp,axis = 0)

coord = a1b.coord('time')
dt = coord.units.num2date(coord.points)
year = np.array([coord.units.num2date(value).year for value in coord.points])

high_amo_loc = np.where((year >= 1930) & (year < 1950))[0]
low_amo_loc = np.where((year >= 1970) & (year < 1980))[0]

high_amo1 = a1b[high_amo_loc].collapsed('time',iris.analysis.MEAN)
low_amo1 = a1b[low_amo_loc].collapsed('time',iris.analysis.MEAN)

'''

plt.close('all')
qplt.contourf((low_amo1-high_amo1)-(low_amo-high_amo),np.linspace(-1.5,1.5,31))
plt.gca().coastlines()
#plt.show()
plt.title('tas')
plt.savefig('/home/ph290/Documents/figures/hadgem_tas.png')
