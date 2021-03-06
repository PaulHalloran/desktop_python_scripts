from iris.coords import DimCoord
import iris.plot as iplt
import time
import numpy as np
import glob
import iris
import iris.coord_categorisation
import iris.analysis
import subprocess
import os
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy.ma as ma
import running_mean as rm
import running_mean_post as rmp
from scipy import signal
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import running_mean_post
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import iris.analysis.cartography
import numpy.ma as ma
import scipy.interpolate
import gc
import pickle
import biggus
import seawater
import cartopy.feature as cfeature

f = open('/home/ph290/Documents/logs/palaeo_amo_amoc_paper_figures_log.txt','w')


'''
Note - all model data regridded bilinearly on to 1x1 degree unless velocity, which is gridded to 1.4 degree
Figure 1:
Models used for AMOC and tas analysis: print all_models
Steps:
1) Regrid v-velocity data to 1/4 degree horizontally (bilinear interpolation), but keeping the same vertical levels (done in other script)
2) Create a mask of the Atlantic (1st idem below)
3) Makeing the different models, calculate the maximum overturning stream function for those models at 26N (and 45N) (collapse along longitudes, then do cumulative sum top to bottom - or is it vice cerse, should not matter)
CHECK MASK IS CORRECT WIUTH NEW RESOLUTOIN!
     this is held in 'max_strm_fun_26'
4) Read in model tas and area average this across the AMO box
5) Read in Mann AMO data
   	- select just the years between 850 and 1850
   	- high-pass filter  to remove variability with a period longer than 100 year
	- subtract the min value and divide by the max minus the min (i.e. converting all to a range from 0 to 1)
6) Read in crowley unterman volcanic aod data for N and S. hemisphere respectively
7) Using all models (but just 1st GISS ens. member (1st ensemble forcing) - NO NOT USING GIS'COS TERRIBLE AMOC DRIFT)
	- loop through models:
		- select just the years between 850 and 1850
		- high-pass filter  to remove variability with a period longer than 100 year
		- subtract the min value and divide by the max minus the min (i.e. converting all to a range from 0 to 1)
		NO- apply a running mean with a smoothing of 5 years ('smoothing_val')
		- apply 3 year low-pass filter
		- plot as 'CMIP5/PMIP3 ensemble member'
		- mean across ensemble members to plot as 'CMIP5/PMIP3 ensemble mean'
#DON't do this if I can doload the remeiniug data - trying now... 7b) remove FGOALS-gl from the model list because this does not start in the year 850, it starts in the year 1000 and because other FGOALS model does
#DON't do this if I can dowload the remeiniug data - trying now... 7c) MRI-CGCM3 removed from the model list because stream function timeseries too short. Look into this was something wrong with input files?
GISS-E2-R ALSO REMOVED BECAUSE IT's stream fuction  is massively drifting
Other FGOALES also removed
#NOTE so_Omon_bcc-csm1-1_past1000_r1i1p1_119001-119912.nc is missing and not on the CMIP5 archive. This causes problems

8) Extract 850 to 1850 years (check why this is not 850-1850) from the Crowley volcanic index
	- Apply a 7-year running mean to this data (identified in a monticarlo analysis as having the best explanatory power of the SSTs)
	- Interpolate multi-model mean PMIP3 on to the approx. daily timescale of the volcanic data and produce an ordinary least squares linear model explaining this SST with the smoothed volcanic data
	- plot the tas and volcanic-based model of tas
9) Using all models (but just 1st GISS ens. member (1st ensemble forcing))
	- loop through models:
		- select just the years between 850 and 1850
		- high-pass filter  to remove variability with a period longer than 100 year
		- subtract the min value and divide by the max minus the min (i.e. converting all to a range from 0 to 1)
		- apply a running mean with a smoothing of 5 years ('smoothing_val') 
		- plot as 'CMIP5/PMIP3 ensemble member'
		- mean across ensemble members to plot as 'CMIP5/PMIP3 ensemble mean', smooth (smoothing of 5 years - why not 10? - smoothing_val = 5) and plot
		- NOTE: we are normalising before meaning together. This is why the data does not fill the whole 0-1 range

Figure 2:
- Identify which models have tos, sos and precipitation - this model list is called: models_tas_sos_pr
- Define N. Iceland region:
west = -24
east = -13
south = 65
north = 67
- Sequentially read in model tas, sos and pr data
	- Extract above identified region and area average
	- convert temperature from kelvin to Celsius
	- Calculate surface ocean density in three ways:
		- using raw t and s
		- holding T at its mean value throughout the first 1000 years of the run (because some run on to 2005)
		- holding salinity at its mean value from throughout the first 1000 years of the run (because some run on to 2005)
	- Accumulate all of the data from each iteration (including the non-density data) into a dictionary called 'density_data'
- Looping through this dictionary I:
	- high-pass filter (100 years)
	- Normalise non-density data (take of mean and divide by range) - note that the script did not originally do this, so if problems, examine this part of script
	- collapse models together to produce multi-model means
-plotting:
- Panel 1:
	- With a smoothing window of 5 years...
	- Read in Reynolds d180 and high-pass filter (100 years)
	- perform running mean (5-year - why not longer? - 'tmp = rm.running_mean(tmp,smoothing_val)')
	- Plot Reynolds d18O
	- Plot tas as in figure 1
- Panel 2:
	- Plot AMOC as in figure 1
	- plot multi-model mean density (currently normalised, but might be best not to have this)
- Panel 3:
	- plot multi model mean density, then the same thing but calculated with constant temperature and constant salinity
- Panel 4:
	- plotting multi-model salinity and precipitation and N. hem. volcanoes

Figure 3:
#ERA NAO
- Read in ERA-interim 'moisture flux'
- high-pass filtering this data long the time axis (note this was not done in the previous script)
- Read in winter NAO from http://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/JFM_season_nao_index.shtml
- high-pass filtering this data long the time axis (note this was not done in the previous script)
- Subtract the mean from the NAO index
- identify where the NAO index is above and below zero
- Average together the ERA moisture flux from all of the high NAO years and all of the low NAO years
- Extract the high-latitude region for plotting
	- NOTE - COULD HIGH PASS FILTER THE NAO AND ERA DATA?
#PMIP3 salinity/precip
- Using the multi-model mean precip. timeseries calculated for N. Iceland
	- avoid the each-end 100 years (to avoid problems with the high-pass filter)
	- read in the precip. from each model sequentially
		- high-pass filter in the time direction
		- Identify the years that correspond to high and low N. Iceland salinity
		- Mean these two sets of years independently
		- Put into 3-D arrays (model-lat-lon)
	- mean together 3-D array along the 'model' axis
	- Extract the polar(ish) region, holding results in pr_cube_high_mean and pr_cube_low_mean
#PMIP3 volcanoes/precip
	- Almost exactly as above, but using volcanic index rather than salinity timeseries
		- differences:
			- (still avoiding 100 years at each end)
			- Taking volcanic years that are above or below the median to give a reasonable number
#Plotting:
- Plotting low minus high for PMIP3
- Plotting high minus low for ERA - check sign (remember moisture flux rather than precip. Can we do this a better way?)
			

NOTE: bcc-csm1-1 has a jump in stream function after about 350 years - sort out...
Either sort out of excluse form analysis. Exlcusing to start with....



###
#Filter
###

# N=5.0
# #N is the order of the filter - i.e. quadratic
# timestep_between_values=1.0 #years value should be '1.0/12.0'
# low_cutoff=100.0
# 
# Wn_low=timestep_between_values/low_cutoff
# 
# b, a = scipy.signal.butter(N, Wn_low, btype='high')

'''

def extract_years(cube):
	try:
		iris.coord_categorisation.add_year(cube, 'time', name='year2')
	except:
		'already has year2'
	loc = np.where((cube.coord('year2').points >= start_year) & (cube.coord('year2').points <= end_year))
	loc2 = cube.coord('time').points[loc[0][-1]]
	cube = cube.extract(iris.Constraint(time = lambda time_tmp: time_tmp <= loc2))
	return cube

def butter_lowpass(lowcut, fs, order=5):
    nyq = fs
    low = lowcut/nyq
    b, a = scipy.signal.butter(order, low , btype='high',analog = False)
    return b, a



def butter_highpass(highcut, fs, order=5):
    nyq = fs
    high = highcut/nyq
    b, a = scipy.signal.butter(order, high , btype='low',analog = False)
    return b, a


def butter_bandpass(lowcut, fs, order=5):
    nyq = fs
    low = lowcut/nyq
    b, a = scipy.signal.butter(order, low , btype='high',analog = False)
    return b, a
    
#b, a = butter_bandpass(1.0/100.0, 1)
'''

# N=5.0
# #N is the order of the filter - i.e. quadratic
# timestep_between_values=1.0 #years value should be '1.0/12.0'
# low_cutoff=100.0
# 
# Wn_low=low_cutoff/(timestep_between_values * 0.5)
# 
# b, a = scipy.signal.butter(N, Wn_low, btype='high')

'''
#producing an Atlantic mask (mask masked and Atlantic has value of 1, elsewhere zero) to use in the stream function calculation
'''

#input_file = '/data/temp/ph290/regridded/cmip5/last1000_vo_amoc_high_res/CCSM4_vo_past1000_r1i1p1_regridded_not_vertically.nc'
input_file = '/media/usb_external1/tmp/MRI-CGCM3_vo_past1000_r1i1p1_regridded_not_vertically.nc'
cube = iris.load_cube(input_file)
cube = cube[0,0]
cube.data = ma.masked_where(cube.data == 0,cube.data)
#tmp = cube.lazy_data()
#tmp = biggus.ma.masked_where(tmp.ndarray() == 0,tmp.masked_array())

resolution = 0.25

start_date = 850
end_date = 1850

tmp_cube = cube.copy()
tmp_cube = tmp_cube*0.0

location = -30/resolution

print 'masking forwards'

for y in np.arange(180/resolution):
    print 'lat: ',y,'of ',180/resolution
    flag = 0
    tmp = tmp_cube.data.mask[y,:]
    tmp2 = tmp_cube.data[y,:]
    for x in np.arange(360/resolution):
        if tmp[location] == True:
            flag = 1
        if ((tmp[location] == False) & (flag == 0)):
            tmp2[location] = 1
        tmp = np.roll(tmp,+1)
        tmp2 = np.roll(tmp2,+1)
    tmp_cube.data.mask[y,:] = tmp
    tmp_cube.data.data[y,:] = tmp2.data

location = location+1

print 'masking backwards'

for y in np.arange(180/resolution):
    print 'lon: ',y,'of ',180/resolution
    flag = 0
    tmp = tmp_cube.data.mask[y,:]
    tmp2 = tmp_cube.data[y,:]
    for x in np.arange(360/resolution):
        if tmp[location] == True:
            flag = 1
        if ((tmp[location] == False) & (flag == 0)):
            tmp2[location] = 1
        tmp = np.roll(tmp,-1)
        tmp2 = np.roll(tmp2,-1)
    tmp_cube.data.mask[y,:] = tmp
    tmp_cube.data.data[y,:] = tmp2.data

tmp_cube.data.data[150/resolution:180/resolution,:] = 0.0
tmp_cube.data.data[0:40/resolution,:] = 0.0
tmp_cube.data.data[:,20/resolution:180/resolution] = 0.0
tmp_cube.data.data[:,180/resolution:280/resolution] = 0.0

loc = np.where(tmp_cube.data.data == 0.0)
tmp_cube.data.mask[loc] = True

mask1 = tmp_cube.data.mask
cube_test = []

f.write('maske created\n')

'''
#calculating stream function
'''

#trying with the 1/4 degree dataset rather than the 1x1 - this should make the stram function calculatoi nmore robust
files = glob.glob('/data/temp/ph290/regridded/cmip5/last1000_vo_amoc_high_res/*_vo_*.nc')
#/media/usb_external1/cmip5/last1000_vo_amoc



models = []
max_strm_fun = []
max_strm_fun_26 = []
max_strm_fun_45 = []
model_years = []

f.write('files working with:\n')
f.write(str(files)+'\n')

for file in files:

    model = file.split('/')[7].split('_')[0]
    print model
    f.write('processing: '+model+'\n')
    models.append(model)
    cube = iris.load_cube(file)

    print 'applying mask'

    try:
                    levels =  np.arange(cube.coord('depth').points.size)
    except:
                    levels = np.arange(cube.coord('ocean sigma over z coordinate').points.size)

	#for level in levels:
#		print 'level: '+str(level)
#		for year in np.arange(cube.coord('time').points.size):
#			#print 'year: '+str(year)
#			tmp = cube.lazy_data()
#			mask2 = tmp[year,level,:,:].masked_array().mask
#			tmp_mask = np.ma.mask_or(mask1, mask2)
#			tmp[year,level,:,:].masked_array().mask = tmp_mask

    #variable to hold data from first year of each model to check
    #that the maskls have been applied appropriately

    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube[0])
    grid_areas = np.sqrt(grid_areas)

    shape = np.shape(cube)
    tmp = cube[0].copy()
    tmp.data = ma.masked_where(tmp.data == 0,tmp.data)
    tmp = tmp.collapsed('longitude',iris.analysis.SUM)
    collapsed_data = np.tile(tmp.data,[shape[0],1,1])

    mask_cube = cube[0].copy()
    tmp_mask = np.tile(mask1,[shape[1],1,1])
    mask_cube.data.mask = tmp_mask
    mask_cube.data.mask[np.where(mask_cube.data.data == mask_cube.data.fill_value)] = True

    print 'collapsing cube along longitude'
    try:
            slices = cube.slices(['depth', 'latitude','longitude'])
    except:
            slices = cube.slices(['ocean sigma over z coordinate', 'latitude','longitude'])
    for i,t_slice in enumerate(slices):
            #print 'year:'+str(i)
        tmp = t_slice.copy()
        tmp.data = ma.masked_where(tmp.data == 0,tmp.data)
        tmp *= grid_areas
        mask_cube_II = tmp.data.mask
        tmp.data.mask = mask_cube.data.mask | mask_cube_II
        #if i == 0:
            #plt.close('all')
            #qplt.contourf(tmp[0])
            #plt.savefig('/home/ph290/Documents/figures/test/'+model+'l1.png')
            #plt.close('all')
            #qplt.contourf(tmp[10])
            #plt.savefig('/home/ph290/Documents/figures/test/'+model+'l10.png')

        collapsed_data[i] = tmp.collapsed('longitude',iris.analysis.SUM).data

    try:
            depths = cube.coord('depth').points*-1.0
            bounds = cube.coord('depth').bounds
    except:
            depths = cube.coord('ocean sigma over z coordinate').points*-1.0
            bounds = cube.coord('ocean sigma over z coordinate').bounds	
    thickness = bounds[:,1] - bounds[:,0]
    test = thickness.mean()
    if test > 1:
            thickness = bounds[1:,0] - bounds[0:-1,0]
            thickness = np.append(thickness, thickness[-1])

    thickness = np.flipud(np.rot90(np.tile(thickness,[180/resolution,1])))

    tmp_strm_fun_26 = []
    tmp_strm_fun_45 = []
    tmp_strm_fun = []
    for i in np.arange(np.size(collapsed_data[:,0,0])):
            tmp = collapsed_data[i].copy()
            tmp = tmp*thickness
            tmp = np.cumsum(tmp,axis = 1)
            tmp = tmp*-1.0*1.0e-3
            tmp *= 1029.0 #conversion from m3 to kg
            #tmp = tmp*1.0e-7*0.8 # no idea why I need to do this conversion - check...
            coord = t_slice.coord('latitude').points
            loc = np.where(coord >= 26)[0][0]
            tmp_strm_fun_26 = np.append(tmp_strm_fun_26,np.max(tmp[:,loc]))
            loc = np.where(coord >= 45)[0][0]
            tmp_strm_fun_45 = np.append(tmp_strm_fun_45,np.max(tmp[:,loc]))
            tmp_strm_fun = np.append(tmp_strm_fun,np.max(tmp[:,:]))

    coord = cube.coord('time')
    dt = coord.units.num2date(coord.points)
    years = np.array([coord.units.num2date(value).year for value in coord.points])
    model_years.append(years)

    max_strm_fun_26.append(tmp_strm_fun_26)
    max_strm_fun_45.append(tmp_strm_fun_45)
    max_strm_fun.append(tmp_strm_fun)


f.write('saving output to /home/ph290/Documents/python_scripts/pickles/palaeo_amo_III.pickle\n')
with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_IIII.pickle', 'w') as f:
    pickle.dump([models,max_strm_fun,max_strm_fun_26,max_strm_fun_45,model_years,mask1,files,b,a,input_file,resolution,start_date,end_date,location], f)

f.close()

'''

#######################################
#                                     #
#   start of non stream function bit  #
#                                     #
#######################################



with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_IIII.pickle') as f:    models,max_strm_fun,max_strm_fun_26,max_strm_fun_45,model_years,mask1,files,b,a,input_file,resolution,start_date,end_date,location = pickle.load(f)

b, a = butter_bandpass(1.0/100.0, 1.0,2)

print '- NOTE! FGOALS MODEL LEVELS ARE UPSIDE DOWN - DOES THIS MATTER?, check mask figures to explain'

print 'check masks in /home/ph290/Documents/figures'


###
#read in temperature
###

amo_box_tas = []
model_years_tas = []



for i,model in enumerate(models):
	print 'processing: '+model
	file = glob.glob('/media/usb_external1/cmip5/last1000/'+model+'_tos_past1000_r1i1p1_*.nc')
	cube = iris.load_cube(file)
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
	amo_box_tas.append(ts)
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	years = np.array([coord.units.num2date(value).year for value in coord.points])
	model_years_tas.append(years)



###
#Construct dictionaries containing the models to use and the associated stream function and tas. Note just taking the 1st ensmeble from GISS, which using on of the volc forcings etc. (other 'ensemble' members use different forcings etc.)
###

pmip3_tas = {}
pmip3_str = {}
pmip3_year_tas = {}
pmip3_year_str = {}

giss_test = 0

for i,model in enumerate(models):
	if model == 'GISS-E2-R':
		if giss_test == 0:
			pmip3_tas[model] = amo_box_tas[i].data
			pmip3_str[model] = max_strm_fun_26[i]
			pmip3_year_str[model] = model_years[i]
			pmip3_year_tas[model] = model_years_tas[i]
			giss_test += 1
	if model <> 'GISS-E2-R':
		pmip3_tas[model] = amo_box_tas[i].data
		pmip3_str[model] = max_strm_fun_26[i]
		pmip3_year_str[model] = model_years[i]
		pmip3_year_tas[model] = model_years_tas[i]
	
	

#for i,model in enumerate(models):
#	pmip3_tas[model] = amo_box_tas[i].data
#	pmip3_str[model] = max_strmfun_26[i]
#	pmip3_year_str[model] = model_years[i]
#	pmip3_year_tas[model] = model_years_tas[i]
	
#all_models = np.unique(models+models_unique)



print 'have you finished processing the input files for the two following models. Currently downloading extra data'

#hfjohdjklasdjkl
modles = list(models)
models.remove('FGOALS-gl')
#REMOVE FGOALS-gl BECAUSE STARTS IN yr 1000 not 850
models.remove('FGOALS-s2')
#FGOALS-s2 removed because there is a problem with one of teh humidity files
#models.remove('GISS-E2-R')
#NOTE! FGOALS MODEL LEVELS ARE UPSIDE DOWN - DOES THIS MATTER?, check mask figures to explain'
models.remove('bcc-csm1-1')
#NOTE JUST FIXING SALINITY IN THIS MODEL - REQUIRED AN EXTRA FILE WHICH WAS MISSED OFF CMIP5 ARCHIVE
models = np.array(models)

all_models = models
#added = does this cause problems?

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_II.pickle', 'w') as f:
#    pickle.dump([all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,all_models], f)

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_II.pickle') as f:
#ll_models_tas_sos_pr    all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,all_models = pickle.load(f)

all_years = np.linspace(850,1850,(1851-850))

start_year  = 850
end_year = 1850
expected_years = np.arange(850,1850)



##############################################
#            figure                         #
##############################################




directory = '/media/usb_external1/cmip5/last1000/'
#NOTE so_Omon_bcc-csm1-1_past1000_r1i1p1_119001-119912.nc is missing and not on the CMIP5 archive. This causes problems


west = -24
east = -13
south = 65
north = 67

west1 =  -180
east1 = 180
south1 = 80
north1 = 100

west2 = -25
east2 = -15
south2 = 50
north2 = 60

density_data = {}
#models_tas_sos_pr
for model in models:
	print model
	#evap
	# evap_cube = iris.load_cube(directory+model+'_evspsbl_past1000_r1i1p1*.nc')
	# evap_cube = extract_years(evap_cube)
	# try:
	# 	evap_depths = evap_cube.coord('depth').points
	# 	evap_cube = evap_cube.extract(iris.Constraint(depth = np.min(evap_depths)))
	# except:
	# 	print 'no evap depth coordinate'
	# temporary_cube = evap_cube.intersection(longitude = (west, east))
	# evap_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
	# try:
	# 	evap_cube_n_iceland.coord('latitude').guess_bounds()
	# 	evap_cube_n_iceland.coord('longitude').guess_bounds()
	# except:
	# 	print 'already have bounds'
	# grid_areas = iris.analysis.cartography.area_weights(evap_cube_n_iceland)
	# evap_cube_n_iceland_mean = evap_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
	#temperature
	t_cube = iris.load_cube(directory+model+'_tos_past1000_r1i1p1_*.nc')
	t_cube = extract_years(t_cube)
	try:
		t_depths = t_cube.coord('depth').points
		t_cube = t_cube.extract(iris.Constraint(depth = np.min(t_depths)))
	except:
		print 'no temperature depth coordinate'
	temporary_cube = t_cube.intersection(longitude = (west, east))
	t_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
	try:
		t_cube_n_iceland.coord('latitude').guess_bounds()
		t_cube_n_iceland.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(t_cube_n_iceland)
	t_cube_n_iceland_mean = t_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
	#salinity
	s_cube = iris.load_cube(directory+model+'_sos_past1000_r1i1p1_*.nc')
	s_cube = extract_years(s_cube)
	try:
		s_depths = s_cube.coord('depth').points
		s_cube = s_cube.extract(iris.Constraint(depth = np.min(s_depths)))
	except:
		print 'no salinity depth coordinate'
	temporary_cube = s_cube.intersection(longitude = (west, east))
	s_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
	try:
		s_cube_n_iceland.coord('latitude').guess_bounds()
		s_cube_n_iceland.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(s_cube_n_iceland)
	s_cube_n_iceland_mean = s_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
	#density
	tmp_density = seawater.dens(s_cube_n_iceland_mean,t_cube_n_iceland_mean-273.15)
	test = np.size(t_cube_n_iceland_mean)
	if test <= 1000:
		temp_t_mean = np.mean(t_cube_n_iceland_mean)
	if test > 1000:
		temp_t_mean = np.mean(t_cube_n_iceland_mean[0:1000])
	test = np.size(s_cube_n_iceland_mean)
	if test <= 1000:
		temp_s_mean = np.mean(s_cube_n_iceland_mean)
	if test > 1000:
		temp_s_mean = np.mean(s_cube_n_iceland_mean[0:1000])
	tmp_temp_mean_density = seawater.dens(s_cube_n_iceland_mean,t_cube_n_iceland_mean*0.0+temp_t_mean-273.15)
	tmp_sal_mean_density = seawater.dens(s_cube_n_iceland_mean*0.0+temp_s_mean, t_cube_n_iceland_mean-273.15)
	#years
	coord = t_cube_n_iceland.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	#output
	density_data[model] = {}
	#density_data[model]['evspsbl'] = evap_cube_n_iceland_mean
	density_data[model]['temperature'] = t_cube_n_iceland_mean
	density_data[model]['salinity'] = s_cube_n_iceland_mean
	density_data[model]['density'] = tmp_density
	density_data[model]['temperature_meaned_density'] = tmp_temp_mean_density
	density_data[model]['salinity_meaned_density'] = tmp_sal_mean_density
	density_data[model]['years'] = year_tmp





###
#Averaging the density data
###
	
min_yr = 10000
max_yr = 0
for model in density_data.viewkeys():
	tmp_min_yr = np.min(density_data[model]['years'])
	if 	tmp_min_yr < min_yr:
		min_yr = tmp_min_yr
	tmp_max_yr = np.max(density_data[model]['years'])
	if 	tmp_max_yr > max_yr:
		max_yr = tmp_max_yr
		

mean_density = np.zeros([max_yr+1 - min_yr,len(density_data)])
mean_density[::] = np.NAN
mean_temperature = mean_density.copy()
mean_salinity = mean_density.copy()
temperature_meaned_density = mean_density.copy()
salinity_meaned_density = mean_density.copy()
#evap = mean_density.copy()

smoothing_val = 5

b1, a1 = butter_lowpass(1.0/100.0, 1.0,2)
b2, a2 = butter_highpass(1.0/3, 1.0,2)


years = range(min_yr,max_yr+1)
for i,model in enumerate(density_data.viewkeys()):
	tmp_yrs = density_data[model]['years']
	data1 = density_data[model]['density']
	#data1 = scipy.signal.filtfilt(b, a, data1)
	data1 = scipy.signal.filtfilt(b1, a1, data1)
	data1 = scipy.signal.filtfilt(b2, a2, data1)
	#data1 = rm.running_mean(data1,smoothing_val)
	x = data1
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data1 = x
	data2 = density_data[model]['temperature']
	#data2 = scipy.signal.filtfilt(b, a, data2)
	data2 = scipy.signal.filtfilt(b1, a1, data2)
	data2 = scipy.signal.filtfilt(b2, a2, data2)
	#data2 = rm.running_mean(data2,smoothing_val)
	x = data2
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data2 = x
	data3 = density_data[model]['salinity']
	#data3 = scipy.signal.filtfilt(b, a, data3)
	data3 = scipy.signal.filtfilt(b1, a1, data3)
	data3 = scipy.signal.filtfilt(b2, a2, data3)
	#data3 = rm.running_mean(data3,smoothing_val)
	x = data3
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data3 = x
	data4 = density_data[model]['temperature_meaned_density']
# 	data4 = scipy.signal.filtfilt(b, a, data4)
	data4 = scipy.signal.filtfilt(b1, a1, data4)
	data4 = scipy.signal.filtfilt(b2, a2, data4)
	#data4 = rm.running_mean(data4,smoothing_val)
	x = data4
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data4 = x
	data5 = density_data[model]['salinity_meaned_density']
# 	data5 = scipy.signal.filtfilt(b, a, data5)
	data5 = scipy.signal.filtfilt(b1, a1, data5)
	data5 = scipy.signal.filtfilt(b2, a2, data5)
	#data5 = rm.running_mean(data5,smoothing_val)
	x = data5
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data5 = x
	# data10 = density_data[model]['evspsbl']
	# data10 = scipy.signal.filtfilt(b, a, data10)
	# data10 = rm.running_mean(data10,smoothing_val)
	# x = data10
	# x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	# data10 = x
	#assigning this data to 2-D (model/time) array for meaning
	for j,tmp_yr in enumerate(tmp_yrs):
		loc = np.where(tmp_yr == years)
		mean_density[loc,i] = data1[j]
		mean_temperature[loc,i] = data2[j]
		mean_salinity[loc,i] = data3[j]
		temperature_meaned_density[loc,i] = data4[j]
		salinity_meaned_density[loc,i] = data5[j]
                #evap[loc,i] = data10[j]





mean_density = np.ma.masked_invalid(mean_density)
mean_density2 = np.ma.mean(mean_density,axis = 1)
mean_temperature = np.ma.masked_invalid(mean_temperature)
mean_temperature2 = np.ma.mean(mean_temperature,axis = 1)
mean_salinity = np.ma.masked_invalid(mean_salinity)
mean_salinity2 = np.ma.mean(mean_salinity,axis = 1)
temperature_meaned_density = np.ma.masked_invalid(temperature_meaned_density)
temperature_meaned_density2 = np.ma.mean(temperature_meaned_density,axis = 1)
salinity_meaned_density = np.ma.masked_invalid(salinity_meaned_density)
salinity_meaned_density2 = np.ma.mean(salinity_meaned_density,axis = 1)
#evap2 = np.ma.mean(evap,axis = 1)

pmip3_model_streamfunction = np.zeros([1+end_year-start_year,np.size(all_models)])
pmip3_model_streamfunction[:] = np.NAN

smoothing_val = 5

for i,model in enumerate(models):
        print model
        tmp = pmip3_str[model]
#       tmp = rm.running_mean(tmp,smoothing_val)
        loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
        tmp = tmp[loc]
        yrs = pmip3_year_str[model][loc]
#         data2 = scipy.signal.filtfilt(b, a, tmp)
        data2 = scipy.signal.filtfilt(b1, a1, tmp)
        data2 = scipy.signal.filtfilt(b2, a2, data2)
        #data2 = tmp
        x = data2
        data3 = (x-np.min(x))/(np.max(x)-np.min(x))
        data3 = data3-np.mean(data3)
#         data3 = rm.running_mean(data3,smoothing_val)
        for index,y in enumerate(expected_years):
            loc2 = np.where(yrs == y)
            if np.size(loc2) != 0:
                pmip3_model_streamfunction[index,i] = data3[loc2]

        # for j,yr_tmp in enumerate(range(start_year,end_year)):
        #       loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
        #       if np.size(loc2) > 0:
        #               pmip3_model_streamfunction[j,i] = data3[loc2]

pmip3_multimodel_mean_streamfunction = np.mean(pmip3_model_streamfunction, axis = 1)



###
#Start plotting
###

smoothing_val = 5
alph = 0.2
wdth = 2


start_date = 850
end_date = 1850



plt.close('all')
fig = plt.figure(figsize=(10,10))

###
#Top Panel
###

ax21 = fig.add_subplot(311)

#for i,dummy in enumerate(models):
#all_models_tas_sos_pr):
#	l21a = ax21.plot(range(start_year,end_year+1),rm.running_mean(pmip3_model_streamfunction[:,i],smoothing_val),'g',alpha = 0.1,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble member AMOC')

'''
# l21b = ax21.plot(yrs,pmip3_multimodel_mean_streamfunction,'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean AMOC')
# ax21.set_ylim([-0.15,0.15])
# ax21.set_ylabel('Normalised\nAMOC strength')
'''

r_data_file = '/home/ph290/data0/reynolds/ultra_data.csv'
r_data = np.genfromtxt(r_data_file,skip_header = 1,delimiter = ',')
tmp = r_data[:,1]
# tmp = scipy.signal.filtfilt(b, a, tmp)
tmp = scipy.signal.filtfilt(b1, a1, tmp)
tmp = scipy.signal.filtfilt(b2, a2, tmp)
# tmp = rm.running_mean(tmp,smoothing_val)
loc = np.where((np.logical_not(np.isnan(tmp))) & (r_data[:,0] >= start_date) & (r_data[:,0] <= end_date))
tmp = tmp[loc]
tmp_yr = r_data[loc[0],0]
l21b = ax21.plot(tmp_yr,tmp,'r',linewidth = 2,alpha = 0.75,label = 'Reynolds et al. (2014) $\delta^{18}$O')
ax21.set_ylabel('$\delta^{18}$O')

ax22 = ax21.twinx()
#l22 = ax22.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')
l22 = ax22.plot(years,mean_density2,'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')
ax22.set_ylabel('Density anomaly\n(kg/m$^3$)')

'''
common_year1 = np.in1d(tmp_yr,years)
common_year2 = np.in1d(years,tmp_yr)
years = np.array(years)
x = tmp[common_year1]
y = mean_density2[common_year2]

fig = plt.figure(figsize=(10,10))
axa = fig.add_subplot(111)
axa.plot(tmp_yr[common_year1],x,'b')
axb = axa.twinx()
axb.plot(years[common_year2],y,'r')
plt.show()

fig = plt.figure(figsize=(10,10))
axa = fig.add_subplot(111)
axa.plot(x[::-1],'b')
axb = axa.twinx()
axb.plot(y,'r')
plt.show()

b1b, a1b = butter_lowpass(1.0/90, 1.0,2)
b2b, a2b = butter_highpass(1.0/10, 1.0,2)

a = x[::-1]
b = y

a = a[300::]
b = b[300::]

a = scipy.signal.filtfilt(b1b, a1b, a)
b = scipy.signal.filtfilt(b1b, a1b, b)
a = scipy.signal.filtfilt(b2b, a2b, a)
b = scipy.signal.filtfilt(b2b, a2b, b)
a = a[np.logical_not(np.isnan(a))]
b = b[np.logical_not(np.isnan(b))]
scipy.stats.pearsonr(a,b)
fig = plt.figure(figsize=(10,5))
axa = fig.add_subplot(111)
axa.plot(a,'b')
axb = axa.twinx()
axb.plot(b,'r')
plt.show()
'''

ax21.set_xlim([950,1850])

lns = l21b+l22 
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':12})

###
#second panel down (AMOC and density)
###

ax11 = fig.add_subplot(312)


#l11 = ax11.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')
l11 = ax11.plot(years,mean_density2,'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')

ax12 = ax11.twinx()
#tmp2 = rm.running_mean(pmip3_multimodel_mean_streamfunction,smoothing_val)
tmp_yr = years


l12 = ax12.plot(years,pmip3_multimodel_mean_streamfunction,'g',linewidth = 2,alpha = 0.75,label = 'CMIP5/PMIP3 ensemble mean AMOC')

#ax11.set_ylim([-2.0,2.0])
ax11.set_xlim([950,1850])
#ax11.set_ylabel('$\delta^{18}$O')
ax11.set_ylabel('Density anomaly\n(kg/m$^3$)')
ax12.set_ylabel('Normalised\nAMOC strength')

lns = l11+l12
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':12})



###
#3rd panel down (what is driving density?)
###

ax31 = fig.add_subplot(313)
#l31 = ax31.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland density - note filtering poss. messing up ends')
l31 = ax31.plot(years,mean_density2,'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland density')
#l31b = ax31.plot(years,rm.running_mean(temperature_meaned_density2,smoothing_val),'r',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to salinity)')
l31b = ax31.plot(years,temperature_meaned_density2,'r',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to salinity)')
#l31c = ax31.plot(years,rm.running_mean(salinity_meaned_density2,smoothing_val),'b',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to temperature)')
l31c = ax31.plot(years,salinity_meaned_density2,'b',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to temperature)')

ax31.set_ylabel('Density anomaly\n(kg/m$^3$)')
ax31.set_xlabel('Calendar Year')

lns = l31+l31b+l31c
 
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':12})


ax11.set_xlim([950,1850])
ax21.set_xlim([950,1850])
ax31.set_xlim([950,1850])

#plt.show(block = True)
plt.savefig('/home/ph290/Documents/figures/palaeoamo/figure1_April15f.png')
