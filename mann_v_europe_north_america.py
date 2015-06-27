import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import running_mean as rm
import iris
import iris.quickplot as qplt
import scipy
import scipy.stats

# europea = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 96,skip_footer = 7183-1504-23)
# europea_lons = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 92,usecols = np.arange(25)+1,skip_footer = 7183-23-93)
# europea_lats = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 93,usecols = np.arange(25)+1,skip_footer = 7183-23-94)
# 
# europeb = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 1516,skip_footer = 7183-2924-17)
# europeb_lons = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 1512,usecols = np.arange(25)+1,skip_footer = 7183-17-1513)
# europeb_lats = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 1513,usecols = np.arange(25)+1,skip_footer = 7183-17-1514)
# 
# europec = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 2935,skip_footer = 7183-4343-12)
# europec_lons = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 2931,usecols = np.arange(25)+1,skip_footer = 7183-12-2932)
# europec_lats = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 2932,usecols = np.arange(25)+1,skip_footer = 7183-12-2933)
# 
# europed = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 4354,skip_footer = 7183-5762-7)
# europed_lons = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 4350,usecols = np.arange(25)+1,skip_footer = 7183-7-4351)
# europed_lats = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 4351,usecols = np.arange(25)+1,skip_footer = 7183-7-4352)
# 
# europee = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 5772,skip_footer = 7183-7180-4)
# europee_lons = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 5768,usecols = np.arange(25)+1,skip_footer = 7183-4-5768)
# europee_lats = np.genfromtxt('/home/ph290/data0/misc_data/guiot2010europe.txt',skip_header = 5769,usecols = np.arange(25)+1,skip_footer = 7183-4-5769)
# #ftp://ftp.ncdc.noaa.gov/pub/data/paleo/contributions_by_author/guiot2010/guiot2010europe.txt
# 
# europe_yr = europea[:,0]
# no_years = np.size(europe_yr) 
# 
# latitude = iris.coords.DimCoord(range(-90, 90, 5), standard_name='latitude', units='degrees')
# longitude = iris.coords.DimCoord(range(-180, 180, 5), standard_name='longitude', units='degrees')
# time = iris.coords.DimCoord(range(int(np.min(europe_yr)),int(np.max(europe_yr)+1 ),1), standard_name='time', units='year')
# 
# cube_data = np.ma.zeros((no_years,180/5, 360/5), np.float32)
# 
# latitude = iris.coords.DimCoord(np.arange(27.5, 72.5+5, 5), standard_name='latitude', units='degrees')
# longitude = iris.coords.DimCoord(np.arange(-7.5, 57.5+5, 5), standard_name='longitude', units='degrees')
# time = iris.coords.DimCoord(range(int(np.min(europe_yr)),int(np.max(europe_yr)+1 ),1), standard_name='time', units='year')
# 
# cube_data = np.ma.zeros((no_years,10, 14), np.float32)
# 
# cube_data.mask = True
# 
# cube = iris.cube.Cube(cube_data,standard_name='surface_temperature', long_name='Surface Air Temperature', var_name='tas', units='K',dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)])
# c_lons = cube.coord('longitude').points
# c_lats = cube.coord('latitude').points
# 
# for i,dummy in enumerate(europea_lons):
# 	loc1 = np.where(c_lons == europea_lons[i])[0]
# 	loc2 = np.where(c_lats == europea_lats[i])[0]
# 	cube_data.data[:,loc2[0],loc1[0]] = europea[:,i+1]
# 	cube_data.mask[:,loc2[0],loc1[0]] = False
# 	loc1 = np.where(c_lons == europeb_lons[i])[0]
# 	loc2 = np.where(c_lats == europeb_lats[i])[0]
# 	cube_data.data[:,loc2[0],loc1[0]] = europeb[:,i+1]
# 	cube_data.mask[:,loc2[0],loc1[0]] = False
# 	loc1 = np.where(c_lons == europec_lons[i])[0]
# 	loc2 = np.where(c_lats == europec_lats[i])[0]
# 	cube_data.data[:,loc2[0],loc1[0]] = europec[:,i+1]
# 	cube_data.mask[:,loc2[0],loc1[0]] = False
# 	loc1 = np.where(c_lons == europed_lons[i])[0]
# 	loc2 = np.where(c_lats == europed_lats[i])[0]
# 	cube_data.data[:,loc2[0],loc1[0]] = europed[:,i+1]
# 	cube_data.mask[:,loc2[0],loc1[0]] = False
# 	loc1 = np.where(c_lons == europee_lons[i])[0]
# 	loc2 = np.where(c_lats == europee_lats[i])[0]
# 	cube_data.data[:,loc2[0],loc1[0]] = europee[:,i+1]
# 	cube_data.mask[:,loc2[0],loc1[0]] = False
# 	
# cube = iris.cube.Cube(cube_data,standard_name='surface_temperature', long_name='Surface Air Temperature', var_name='tas', units='K',dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)])
# cube.coord('latitude').guess_bounds()
# cube.coord('longitude').guess_bounds()
# 
# lon_west = 20.0
# lon_east = 40.0
# lat_south = 45.0
# lat_north = 60.0 
# 
# cube_region_tmp = cube.intersection(longitude=(lon_west, lon_east))
# cube_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))
# 
# 
# #plt.close('all')
# #qplt.contourf(cube[1000])
# #plt.gca().coastlines()
# #plt.show(block=False)
# 
# 
# grid_areas = iris.analysis.cartography.area_weights(cube_region)
# mean_europe_data = cube_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas).data

N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
middle_cuttoff_high=100.0

Wn_mid_high=timestep_between_values/middle_cuttoff_high

b, a = scipy.signal.butter(N, Wn_mid_high, btype='high')
#scipy.signal.filtfilt(b, a, )

# loc = np.where((europe_yr <= 1850) & (europe_yr >= 1040))
# europe_yr = europe_yr[loc[0]]
# mean_europe_data = mean_europe_data[loc[0]]
# mean_europe_data = scipy.signal.filtfilt(b, a, mean_europe_data)


e_europe = np.genfromtxt('/home/ph290/data0/misc_data/tatra2013temp.txt',skip_header = 90)
#ftp://ftp.ncdc.noaa.gov/pub/data/paleo/treering/reconstructions/europe/tatra2013temp.txt
e_europe_yr = e_europe[:,0]
loc = np.where((e_europe_yr <= 1850) & (e_europe_yr >= 1040))
e_europe_yr = e_europe_yr[loc]
e_europe_data = e_europe[loc[0],1]
e_europe_data = scipy.signal.filtfilt(b, a, e_europe_data)
x = e_europe_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
e_europe_data = x

c_usa =  np.genfromtxt('/home/ph290/data0/misc_data/colorado-plateau2005.txt',skip_header = 79)
#ftp://ftp.ncdc.noaa.gov/pub/data/paleo/treering/reconstructions/northamerica/usa/colorado-plateau2005.txt
c_usa_yr = c_usa[:,0]
loc = np.where((c_usa_yr <= 1850) & (c_usa_yr >= 1040))
c_usa_yr = c_usa_yr[loc]
c_usa_data = c_usa[loc[0],1]
c_usa_data = scipy.signal.filtfilt(b, a, c_usa_data)
x = c_usa_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
c_usa_data = x

# greenland = np.genfromtxt('/home/ph290/data0/misc_data/gisp2-ar-n2-temperature2010.txt',skip_header = 227,usecols = [0,1],skip_footer = 20205-1222-58)
#ftp://ftp.ncdc.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/isotopes/gisp2-ar-n2-temperature2010.txt
# greenland_yr = greenland[:,0]# 
# loc = np.where((greenland_yr <= 1850) & (greenland_yr >= 1040))
# greenland_yr = greenland_yr[loc]
# greenland_data = greenland[loc[0],1]
# greenland_data = scipy.signal.filtfilt(b, a, greenland_data)
# greenland_data = signal.detrend(greenland_data)

east_canada = np.genfromtxt('/home/ph290/data0/misc_data/east-canada2014temp.txt',skip_header = 112,usecols = [0,1])
# ftp://ftp.ncdc.noaa.gov/pub/data/paleo/treering/reconstructions/northamerica/canada/east-canada2014temp.txt
east_canada_yr = east_canada[:,0]
loc = np.where((east_canada_yr <= 1850) & (east_canada_yr >= 1040))
east_canada_yr = east_canada_yr[loc]
east_canada_data = east_canada[loc[0],1]
east_canada_data = scipy.signal.filtfilt(b, a, east_canada_data)
x = east_canada_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
east_canada_data = x

reynolds = np.genfromtxt('/home/ph290/data0/reynolds/ultra_data.csv',skip_header = 1,usecols = [0,1],delimiter=',')
reynolds_yr = reynolds[:,0]
loc = np.where((reynolds_yr <= 1850) & (reynolds_yr >= 1040))
reynolds_yr = reynolds_yr[loc]
reynolds_data = reynolds[loc[0],1]
reynolds_data = scipy.signal.filtfilt(b, a, reynolds_data)
x = reynolds_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
reynolds_data = x

amo_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
amo = np.genfromtxt(amo_file, skip_header = 4)
amo_yr = amo[:,0]
amo_data = amo[:,1]
loc = np.where((amo_yr <= 1850) & (amo_yr >= 1040))
amo_yr = amo_yr[loc]
amo_data = amo_data[loc]
amo_data = scipy.signal.filtfilt(b, a, amo_data)
x = amo_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
amo_data = x

iceland_e_europe_data = np.zeros([np.size(e_europe_data),2])
iceland_e_europe_data[:,0] = e_europe_data
iceland_e_europe_data[:,1] = reynolds_data
iceland_e_europe_data = np.mean(iceland_e_europe_data,axis=1)

colarado_canada_data = np.zeros([np.size(c_usa_data),2])
colarado_canada_data[:,0] = c_usa_data
colarado_canada_data[:,1] = east_canada_data
colarado_canada_data = np.mean(colarado_canada_data,axis = 1)

smoothing = 30
alph_val = 0.75

plt.close('all')

fig = plt.figure(figsize = [12,8])
ax1 = fig.add_subplot(411)
ax1.plot(e_europe_yr,rm.running_mean(e_europe_data,smoothing),'b',linewidth = 3,alpha=alph_val)
ax1.plot(reynolds_yr,rm.running_mean(reynolds_data,smoothing),'b--',linewidth = 3,alpha=alph_val)

ax2 = fig.add_subplot(412)
ax2.plot(c_usa_yr,rm.running_mean(c_usa_data,smoothing),'y',linewidth = 3,alpha=alph_val)
ax2.plot(east_canada_yr,rm.running_mean(east_canada_data,smoothing),'y--',linewidth = 3,alpha=alph_val)

ax3 = fig.add_subplot(413)
#ax3.plot(amo_yr,amo_data,'k',linewidth = 3,alpha=alph_val)
ax4 = ax3.twinx()
ax4.plot(e_europe_yr,rm.running_mean(iceland_e_europe_data,smoothing),'b',linewidth = 3,alpha=alph_val)
ax4.plot(east_canada_yr,rm.running_mean(colarado_canada_data,smoothing),'y',linewidth = 3,alpha=alph_val)

ax4 = fig.add_subplot(414)
ax4.plot(amo_yr,amo_data,'k',linewidth = 3,alpha=alph_val)
ax5 = ax4.twinx()
ax5.plot(e_europe_yr,rm.running_mean(c_usa_data,smoothing)-rm.running_mean(e_europe_data,smoothing),'r',linewidth = 3,alpha=alph_val)

ax1.set_xlim([1000,1850])
ax2.set_xlim([1000,1850])
ax3.set_xlim([1000,1850])
ax4.set_xlim([1000,1850])

#ax4 = ax3.twinx()
#ax4.plot(greenland_yr,greenland_data,'g',linewidth = 3)


plt.show(block = False)
