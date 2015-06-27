import iris
import matplotlib.pyplot as plt
import scipy
import iris.quickplot as qplt
import numpy as np


'''
We will use the following functions, so make sure they are available
'''

def butter_bandpass(lowcut,  cutoff):
    order = 2
    low = 1/lowcut
    b, a = scipy.signal.butter(order, low , btype=cutoff,analog = False)
    return b, a


def low_pass_filter(cube,limit_years):
	b1, a1 = butter_bandpass(limit_years, 'low')
	output = scipy.signal.filtfilt(b1, a1, cube,axis = 0)
	return output


def high_pass_filter(cube,limit_years):
        b1, a1 = butter_bandpass(limit_years, 'high')
        output = scipy.signal.filtfilt(b1, a1, cube,axis = 0)
        return output


'''
Initially just reading in a dataset to work with, and averaging lats and longs to give us a timeseries to plot - you can obviously swap in your timeseries
'''

file = '/media/usb_external1/cmip5/tas_regridded/MPI-ESM-P_tas_piControl_regridded.nc'
cube = iris.load_cube(file)

timeseries1 = cube.collapsed(['latitude','longitude'],iris.analysis.MEAN)

'''
Filtering out everything happening on timescales shorter than than X years (where x is called lower_limit_years)
'''

lower_limit_years = 10.0
output_cube = cube.copy()
output_cube.data = low_pass_filter(cube.data,lower_limit_years)

timeseries2 = output_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN)

plt.close('all')
qplt.plot(timeseries1 - np.mean(timeseries1.data),'r',alpha = 0.5,linewidth = 2)
qplt.plot(timeseries2 - np.mean(timeseries2.data),'g',alpha = 0.5,linewidth = 2)
plt.show(block = True)

'''
Filtering out everything happening on timescales longer than than X years (where x is called upper_limit_years)
'''

upper_limit_years = 5.0
output_cube = cube.copy()
output_cube.data = high_pass_filter(cube.data,upper_limit_years)

timeseries3 = output_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN)

plt.close('all')
qplt.plot(timeseries1 - np.mean(timeseries1.data),'r',alpha = 0.5,linewidth = 2)
qplt.plot(timeseries3 - np.mean(timeseries3.data),'b',alpha = 0.5,linewidth = 2)
plt.show(block = True)

'''
Filtering out everything happening on timescales longer than than X years (where x is called upper_limit_years) but shorter than y years (where y is called lower_limit_years)
'''

upper_limit_years = 50.0
output_cube = cube.copy()
output_cube.data = high_pass_filter(cube.data,upper_limit_years)
lower_limit_years = 5.0
output_cube.data = low_pass_filter(output_cube.data,lower_limit_years)

timeseries4 = output_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN)

plt.close('all')
qplt.plot(timeseries1 - np.mean(timeseries1.data),'r',alpha = 0.5,linewidth = 2)
qplt.plot(timeseries4 - np.mean(timeseries4.data),'y',alpha = 0.5,linewidth = 2)
plt.show(block = True)


'''
Hopefully this tells you everything you need. Just be aware  that strange tings can happen at he ends of the timeseries (just check it is doing something sensible)
'''






