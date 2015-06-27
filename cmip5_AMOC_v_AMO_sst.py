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
import running_mean
import matplotlib.cm as mpl_cm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table


###
#Filter
###

N=5.0
#N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years value should be '1.0/12.0'
low_cutoff=100.0
high_cutoff=10.0

Wn_low=timestep_between_values/low_cutoff
Wn_high=timestep_between_values/high_cutoff

b1, a1 = scipy.signal.butter(N, Wn_low, btype='high')
b2, a2 = scipy.signal.butter(N, Wn_high, btype='low')


directory = '/media/usb_external1/cmip5/msftmyz/piControl/'
directory2 = '/media/usb_external1/cmip5/tas_regridded/'

models = ['CanESM2','CCSM4','CESM1-BGC','CNRM-CM5','MPI-ESM-MR','MRI-CGCM3']
#note FGOALS-g2 had undefined time units, so not using
#don't appear to have tas for NorESM1-M

cmip5_max_strmfun = []
cmip5_strmfun_year = []
cmip5_tas = []
cmip5_tas_year = []

print 'note: filtering out >100 and <10 to identify the sorts of variability we see are looking at in the last millennium analysis' 

for model in models:
	print model+' stream function'
	files = np.array(glob.glob(directory+'*'+model+'_*.nc'))
	cube = iris.load_cube(files)[:,0,:,:]
	loc = np.where(cube.coord('grid_latitude').points >= 26.0)[0]
	lat = cube.coord('grid_latitude').points[loc[0]]
	sub_cube = cube.extract(iris.Constraint(grid_latitude = lat))
	stream_function_tmp = sub_cube.collapsed('depth',iris.analysis.MAX)
	coord = stream_function_tmp.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	#tmp = scipy.signal.filtfilt(b1, a1, stream_function_tmp.data)
	tmp = stream_function_tmp.data
 	tmp = scipy.signal.filtfilt(b2, a2, tmp)
	if not model == 'CNRM-CM5':
		tmp /= 1.0e9
	if model == 'CNRM-CM5':
		tmp /= 1.0e3
# 	tmp = running_mean.running_mean(signal.detrend(stream_function_tmp.data/1.0e9),40)
	#decades = np.unique(np.floor(year_tmp/10.0))
	#tmp2 = []
	#for decade in decades:
	#	loc = np.where(decades == decade)
	#	tmp2.append(np.mean(tmp[loc]))
	#tmp2 = np.array(tmp2)
	cmip5_max_strmfun.append(tmp[np.logical_not(np.isnan(tmp))])
	cmip5_strmfun_year.append(year_tmp[np.logical_not(np.isnan(tmp))])

#cmip5_max_strmfun = np.array(cmip5_max_strmfun)
#cmip5_strmfun_year = np.array(cmip5_strmfun_year)

for model in models:
	print model+' tas'
	files = np.array(glob.glob(directory2+model+'_tas_piControl_regridded.nc'))
	cube = iris.load_cube(files)
	lon_west = -75
	lon_east = -7.5
	lat_south = 0
	lat_north = 60.0
	cube = cube.intersection(longitude=(lon_west, lon_east))
	cube = cube.intersection(latitude=(lat_south, lat_north))
	cube.coord('latitude').guess_bounds()
	cube.coord('longitude').guess_bounds()
	grid_areas = iris.analysis.cartography.area_weights(cube)
	ts = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
	#ts = scipy.signal.filtfilt(b1, a1, ts.data)
	ts = ts.data
 	ts = scipy.signal.filtfilt(b2, a2, ts)
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	#decades = np.unique(np.floor(year_tmp/10.0))
	#ts2 = []
	#for decade in decades:
	#	loc = np.where(decades == decade)
	#	ts2.append(np.mean(ts[loc]))
	#ts2 = np.array(ts2)
	cmip5_tas.append(ts[np.logical_not(np.isnan(ts))])
	cmip5_tas_year.append(year_tmp[np.logical_not(np.isnan(ts))])


#cmip5_tas = np.array(cmip5_tas)
#cmip5_tas_year = np.array(cmip5_tas_year)

cmip5_strmfun_final = []
cmip5_tas_final = []

for i,model in enumerate(models):
	yr1 = cmip5_tas_year[i][:]
	yr2 = cmip5_strmfun_year[i][:]
	common_high_years = np.array(list(set(yr1).intersection(yr2)))
	index1 = np.nonzero(np.in1d(yr1,common_high_years))[0]
	index2 = np.nonzero(np.in1d(yr2,common_high_years))[0]
	cmip5_tas_final.append(cmip5_tas[i][index1])
	cmip5_strmfun_final.append(cmip5_max_strmfun[i][index2])


y = cmip5_tas_final
x = cmip5_strmfun_final


for i,model in enumerate(models):
	x[i][:] = x[i][:]-np.mean(x[i][:])
	y[i][:] = y[i][:]-np.mean(y[i][:])

y = np.concatenate([y[0],y[1],y[2],y[3],y[4],y[5]])
x = np.concatenate([x[0],x[1],x[2],x[3],x[4],x[5]])

xsort = np.argsort(x)
x = x[xsort]
y = y[xsort]

xb = sm.add_constant(x)
model = sm.OLS(y,xb)
results = model.fit()

x2 = np.linspace(np.min(x),np.max(x),np.size(x))
y2 = results.predict(sm.add_constant(x2))

st, data, ss2 = summary_table(results, alpha=0.001)
fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T

plt.scatter(x,y)
#plt.plot(x2,y2,'r')
plt.plot(x, fittedvalues, 'k', lw=2)
#plt.plot(x, predict_ci_low, 'r--', lw=2)
#plt.plot(x, predict_ci_upp, 'r--', lw=2)
plt.plot(x, predict_mean_ci_low, 'r--', lw=2)
plt.plot(x, predict_mean_ci_upp, 'r--', lw=2)
plt.xlabel('Max stream function at 26N (Sv)')
plt.ylabel('AMO-box SST anomaly')
plt.savefig('/home/ph290/Documents/figures/amoc_v_sst.png')
plt.show()


