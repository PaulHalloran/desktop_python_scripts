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
		- apply a running mean with a smoothing of 5 years ('smoothing_val')
		- plot as 'CMIP5/PMIP3 ensemble member'
		- mean across ensemble members to plot as 'CMIP5/PMIP3 ensemble mean'
#DON't do this if I can doload the remeiniug data - trying now... 7b) remove FGOALS-gl from the model list because this does not start in the year 850, it starts in the year 1000 and because other FGOALS model does
#DON't do this if I can dowload the remeiniug data - trying now... 7c) MRI-CGCM3 removed from the model list because stream function timeseries too short. Look into this was something wrong with input files?
GISS-E2-R ALSO REMOVED BECAUSE IT's stream fuction  is massively drifting
Other FGOALES also removed
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
	file = glob.glob('/media/usb_external1/cmip5/tas_regridded/'+model+'_tas_past1000*_regridded.nc')
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
#Read in Mann data
###

amo_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
amo = np.genfromtxt(amo_file, skip_header = 4)
amo_yr = amo[:,0]
amo_data = amo[:,1]
loc = np.where((amo_yr <= end_date) & (amo_yr >= start_date))
amo_yr = amo_yr[loc]
amo_data = amo_data[loc]
amo_data = scipy.signal.filtfilt(b, a, amo_data)
x = amo_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
amo_data = x

###
#read in volc data
###

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)
data4 = np.genfromtxt(file4)

data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]
data = np.mean(data_tmp,axis = 1)
voln_n = data1.copy()
voln_n[:,1] = data

data_tmp[:,0] = data3[:,1]
data_tmp[:,1] = data4[:,1]
data = np.mean(data_tmp,axis = 1)
voln_s = data1.copy()
voln_s[:,1] = data

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
models.remove('GISS-E2-R')
#NOTE! FGOALS MODEL LEVELS ARE UPSIDE DOWN - DOES THIS MATTER?, check mask figures to explain'
#models.remove('bcc-csm1-1')
models = np.array(models)


all_models = models
#added = does this cause problems?

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_II.pickle', 'w') as f:
#    pickle.dump([all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,all_models], f)

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo_II.pickle') as f:
#ll_models_tas_sos_pr    all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,all_models = pickle.load(f)

all_years = np.linspace(850,1850,(1851-850))



##############################################
#            figure 1                        #
##############################################

smoothing_val=10
wdth = 2

plt.close('all')
fig = plt.figure(figsize=(8,12),dpi=80)
ax11 = fig.add_subplot(311)

###
#top panel
###

###
#plot pmip3 surface temperature
###

# start_year = 860
# end_year = 1840

end_year = 1850
start_year = 850
expected_years = np.arange(850,1850)

mean_data = np.zeros([1+end_year-start_year,np.size(all_models)])
mean_data[:] = np.NAN

for i,model in enumerate(all_models):
	tmp = pmip3_tas[model]
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= end_year) & (pmip3_year_tas[model] >= start_year))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b, a, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	l1 = ax11.plot(yrs,rm.running_mean(data3,smoothing_val),'b',alpha = 0.1,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble member')
        for index,y in enumerate(expected_years):
            loc2 = np.where(yrs == y)
            if np.size(loc2) != 0:
                mean_data[index,i] = data3[loc2]
	

mean_data2 = np.mean(mean_data, axis = 1)
l2 = ax11.plot(yrs,mean_data2,'b',linewidth=wdth,alpha=0.9,label = 'CMIP5/PMIP3 ensemble mean')
ax11.set_ylabel('Normalised atlantic temperature anomaly')

###
#plot Mann AMO
###

ax12 = ax11.twinx()
l3 = ax12.plot(amo_yr,amo_data,'k',linewidth=wdth,alpha=0.7,label = 'AMV index (Mann et al., 2009)')
ax12.set_ylim([-0.4,1.2])
ax12.set_ylabel('Normalised AMV index')

lns = l1+l2+l3
 
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.5,prop={'size':8})

###
#middle panel (linear model etc.)
###

ax21 = fig.add_subplot(312)

#smooth volcanic data to 7 years
smth = 7
#start_year = 860
#end_year = 1840
volc_year = voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,0]
vns = running_mean_post.running_mean_post(voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,1],smth*36.0)

###
#produce linear model of tas based on smoothed volcanics
###


y2 = mean_data2.copy()

loc = np.where(np.logical_not(np.isnan(y2)))
y2b = y2[loc]
yrsb = yrs[loc]

model_amo = np.interp(volc_year,yrsb,y2b)


y = model_amo
x1 = vns
x = sm.add_constant(x1)
model = sm.OLS(y,x)
results = model.fit()

###
#plot tas and linear model of tas
###

l1 = ax21.plot(volc_year,y,'b',linewidth=wdth,alpha=0.7,label = 'PMIP3 AMV index')
l2 = ax21.plot(volc_year,results.params[1]*x1+results.params[0],'r',marker ='o', markersize=2,markevery=10*36,linewidth=wdth,alpha=0.7, label = 'statistical model of CMIP5/PMIP3 AMV based on volcanic index')

ax21.set_ylim([0.2,0.9])

###
#plot croweley volcanics
###

ax22 = ax21.twinx()
l3 = ax22.plot(voln_n[:,0],voln_n[:,1],'k',alpha=0.5,label = 'Volcanic index (Crowley and Unterman. 2012)')
ax22.set_label('Aerosol Optical Depth')

lns = l1+l2
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'upper right', fancybox=True, framealpha=0.5,prop={'size':8})

###
#Bottom panel - AMOC
###



pmip3_model_streamfunction = np.zeros([1+end_year-start_year,np.size(all_models)])
pmip3_model_streamfunction[:] = np.NAN

smoothing_val = 5

for i,model in enumerate(models):
	print model
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b, a, tmp)
	#data2 = tmp
        x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	data3 = data3-np.mean(data3)
        data3 = rm.running_mean(data3,smoothing_val)
        for index,y in enumerate(expected_years):
            loc2 = np.where(yrs == y)
            if np.size(loc2) != 0:
                pmip3_model_streamfunction[index,i] = data3[loc2]

	# for j,yr_tmp in enumerate(range(start_year,end_year)):
	# 	loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
	# 	if np.size(loc2) > 0:
	# 		pmip3_model_streamfunction[j,i] = data3[loc2]
	
pmip3_multimodel_mean_streamfunction = np.mean(pmip3_model_streamfunction, axis = 1)

# for i in range(8):
#     print models[i]
#     plt.plot(pmip3_model_streamfunction[:,i])
#     plt.show()



# for model in models:
# 	plt.close('all')
# 	tmp = pmip3_str[model]
# 	#loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
# 	#tmp = tmp[loc]
# 	data2 = scipy.signal.filtfilt(b, a, tmp)
# 	plt.plot(tmp-np.mean(tmp),'r')
# 	plt.plot(data2,'b')
# 	plt.show(block = True)



ax31 = fig.add_subplot(313)

smoothing_val = 5

###
#Plot AMOC (ensemble member then ensemble mean)
###

# for i,dummy in enumerate(all_models):
# 	l1 = ax31.plot(yrs,rm.running_mean(pmip3_model_streamfunction[:,i],smoothing_val),'g',alpha = 0.1,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble member AMOC')
# 
# l2 = ax31.plot(yrs,rm.running_mean(pmip3_multimodel_mean_streamfunction,smoothing_val),'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean AMOC')

for i,dummy in enumerate(all_models):
	l1 = ax31.plot(yrs,pmip3_model_streamfunction[:,i],'g',alpha = 0.1,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble member AMOC')

l2 = ax31.plot(yrs,pmip3_multimodel_mean_streamfunction,'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean AMOC')


ax31.set_ylim([-0.15,0.15])
ax31.set_ylabel('AMOC strength')

###
#Plot Mann AMO index
###

ax32 = ax31.twinx()
l3 = ax32.plot(amo_yr,amo_data,'k',linewidth=wdth,alpha=0.7,label = 'AMV index (Mann et al., 2009)')
ax32.set_ylim([-0.2,1.2])
ax32.set_ylabel('AMV index')

lns = l1+l2+l3
 
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.5,prop={'size':8})


plt.xlim([850,1850])

ax11.set_xlim([950,1850])
ax12.set_xlim([950,1850])
ax21.set_xlim([950,1850])
ax22.set_xlim([950,1850])
ax31.set_xlim([950,1850])
ax32.set_xlim([950,1850])

#plt.show(block = True)
plt.savefig('/home/ph290/Documents/figures/palaeoamo/pmip3_tas_and_stat_modelled_amo_II_hres.png')
	


##############################################
#            figure 2                        #
##############################################




directory = '/media/usb_external1/cmip5/last1000/'

def model_names_tos(directory):
	files = glob.glob(directory+'/*tos*r1i1p1*.nc')
	models_tmp = []
	for file in files:
		statinfo = os.stat(file)
		if statinfo.st_size >= 1:
			models_tmp.append(file.split('/')[-1].split('_')[0])
			models = np.unique(models_tmp)
	return models

def model_names_sos(directory):
        files = glob.glob(directory+'/*sos*r1i1p1*.nc')
        models_tmp = []
        for file in files:
                statinfo = os.stat(file)
                if statinfo.st_size >= 1:
                        models_tmp.append(file.split('/')[-1].split('_')[0])
                        models = np.unique(models_tmp)
        return models

def model_names_pr(directory):
        files = glob.glob(directory+'/*pr*r1i1p1*.nc')
        models_tmp = []
        for file in files:
                statinfo = os.stat(file)
                if statinfo.st_size >= 1:
                        models_tmp.append(file.split('/')[-1].split('_')[0])
                        models = np.unique(models_tmp)
        return models


models_I = model_names_tos(directory)
models_II = model_names_sos(directory)
models_III = model_names_pr(directory)

tmp = list(set(models_I).intersection(models_II))
models_tas_sos_pr = np.array(list(set(tmp).intersection(models_III)))

west = -24
east = -13
south = 65
north = 67

models_tas_sos_pr = list(models_tas_sos_pr)
models_tas_sos_pr.remove('GISS-E2-R')
models_tas_sos_pr = np.array(models_tas_sos_pr)

def extract_years(cube):
	try:
		iris.coord_categorisation.add_year(cube, 'time', name='year2')
	except:
		'already has year2'
	loc = np.where((cube.coord('year2').points >= start_year) & (cube.coord('year2').points <= end_year))
	loc2 = cube.coord('time').points[loc[0][-1]]
	cube = cube.extract(iris.Constraint(time = lambda time_tmp: time_tmp <= loc2))
	return cube

density_data = {}
#models_tas_sos_pr
for model in models_tas_sos_pr:
	print model
# 	try:
	#temperature
	t_cube = iris.load_cube(directory+model+'_tos_past1000_r1i1p1*_regridded_not_vertically.nc')
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
	s_cube = iris.load_cube(directory+model+'_sos_past1000_r1i1p1*_regridded_not_vertically.nc')
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
	#precipitation
	pr_cube = iris.load_cube(directory+model+'_pr_past1000_r1i1p1*_regridded.nc')
	pr_cube = extract_years(pr_cube)
	try:
		pr_depths = pr_cube.coord('depth').points
		pr_cube = pr_cube.extract(iris.Constraint(depth = np.min(pr_depths)))
	except:
		print 'no precipitation depth coordinate'
	temporary_cube = pr_cube.intersection(longitude = (west, east))
	pr_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
	try:
		pr_cube_n_iceland.coord('latitude').guess_bounds()
		pr_cube_n_iceland.coord('longitude').guess_bounds()
	except:
		print 'already have bounds'
	grid_areas = iris.analysis.cartography.area_weights(pr_cube_n_iceland)
	pr_cube_n_iceland_mean = pr_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
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
	density_data[model]['temperature'] = t_cube_n_iceland_mean
	density_data[model]['salinity'] = s_cube_n_iceland_mean
	density_data[model]['density'] = tmp_density
	density_data[model]['temperature_meaned_density'] = tmp_temp_mean_density
	density_data[model]['salinity_meaned_density'] = tmp_sal_mean_density
	density_data[model]['precipitation'] = pr_cube_n_iceland_mean
	density_data[model]['years'] = year_tmp
# 	except:
# 			print 'model can not be read in'


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
precipitation = mean_density.copy()

years = range(min_yr,max_yr+1)
for i,model in enumerate(density_data.viewkeys()):
	tmp_yrs = density_data[model]['years']
	data1 = density_data[model]['density']
	data1 = scipy.signal.filtfilt(b, a, data1)
        data1 = rm.running_mean(data1,smoothing_val)
	data2 = density_data[model]['temperature']
	data2 = scipy.signal.filtfilt(b, a, data2)
        data2 = rm.running_mean(data2,smoothing_val)
	x = data2
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data2 = x
	data3 = density_data[model]['salinity']
	data3 = scipy.signal.filtfilt(b, a, data3)
        data3 = rm.running_mean(data3,smoothing_val)
	x = data3
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data3 = x
	data4 = density_data[model]['temperature_meaned_density']
	data4 = scipy.signal.filtfilt(b, a, data4)
        data4 = rm.running_mean(data4,smoothing_val)
	data5 = density_data[model]['salinity_meaned_density']
	data5 = scipy.signal.filtfilt(b, a, data5)
        data5 = rm.running_mean(data5,smoothing_val)
	data6 = density_data[model]['precipitation']
	data6 = scipy.signal.filtfilt(b, a, data6)
        data6 = rm.running_mean(data6,smoothing_val)
	x = data6
	x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
	data6 = x
	#assigning this data to 2-D (model/time) array for meaning
	for j,tmp_yr in enumerate(tmp_yrs):
		loc = np.where(tmp_yr == years)
		mean_density[loc,i] = data1[j]
		mean_temperature[loc,i] = data2[j]
		mean_salinity[loc,i] = data3[j]
		temperature_meaned_density[loc,i] = data4[j]
		salinity_meaned_density[loc,i] = data5[j]
		precipitation[loc,i] = data6[j]

'''

# def butter_bandpass(lowcut, fs, order=5):
#     nyq = fs
#     low = lowcut/nyq
#     b, a = scipy.signal.butter(order, low , btype='high',analog = False)
#     return b, a
# 
# b, a = butter_bandpass(1.0/100.0, 1,7)
# 
# 
# b, a = scipy.signal.butter(2, 0.01 , btype='high')
# 
# import statsmodels.api as sm
# 
# 
# 
# def high_pass(x,window_len,window='hanning'):
#     if x.ndim != 1:
#         raise ValueError, "smooth only accepts 1 dimension arrays."
#     if x.size < window_len:
#         raise ValueError, "Input vector needs to be bigger than window size."
#     if window_len<3:
#         return x
#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
#     s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
#     #print(len(s))
#     if window == 'flat': #moving average
#         w=np.ones(window_len,'d')
#     else:
#         w=eval('np.'+window+'(window_len)')
#     y=np.convolve(w/w.sum(),s,mode='valid')
#     z = x-y[0:-1.0*(window_len-1)]
#     return z
# 
# 
# import scipy.signal as signal
# n = 100
# a = signal.firwin(n,cutoff = 100,window='hanning')
# a = -a
# 
# def smoothing2(data,win_len):
#     x = smooth(data,window_len=win_len)
#     return data-x[0:-1.0*(win_len-1)]
# 
# data6 = density_data['HadCM3']['precipitation']
# data6b = high_pass(data6,100)
# plt.plot(data6,'r')
# plt.plot(data6b,'b')
# plt.show()
# 
# smoothing_val=50
# 
# for i,model in enumerate(models):
#     data6 = pmip3_str[model]
#     data6 = high_pass(data6,smoothing_val)
#     #data6 = rm.running_mean(data6,smoothing_val)
#     x = data6
#     x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
#     data6 = x
#     data6b = pmip3_str[model]
#     x = data6b
#     x=(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
#     data6b = x
#     print model
#     plt.plot(data6b,'r')
#     plt.plot(data6,'b')
#     plt.show()
# 

'''



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
precipitation = np.ma.masked_invalid(precipitation)
precipitation2 = np.ma.mean(precipitation,axis = 1)

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

ax11 = fig.add_subplot(411)


#l11 = ax11.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')
l11 = ax11.plot(years,mean_density2,'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')

ax12 = ax11.twinx()
#tmp2 = rm.running_mean(pmip3_multimodel_mean_streamfunction,smoothing_val)
tmp_yr = yrs

l12 = ax12.plot(yrs,pmip3_multimodel_mean_streamfunction,'g',linewidth = 2,alpha = 0.75,label = 'CMIP5/PMIP3 ensemble mean AMOC')

#ax11.set_ylim([-2.0,2.0])
ax11.set_xlim([850,1850])
#ax11.set_ylabel('$\delta^{18}$O')
ax11.set_ylabel('Density anomaly\n(kg/m$^3$)')
ax12.set_ylabel('Normalised\nAMOC strength')

lns = l11+l12
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})

###
#second panel down (AMOC and density)
###

ax21 = fig.add_subplot(412)

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
tmp = scipy.signal.filtfilt(b, a, tmp)
tmp = rm.running_mean(tmp,smoothing_val)
loc = np.where((np.logical_not(np.isnan(tmp))) & (r_data[:,0] >= start_date) & (r_data[:,0] <= end_date))
tmp = tmp[loc]
tmp_yr = r_data[loc[0],0]
l21b = ax21.plot(tmp_yr,tmp,'r',linewidth = 2,alpha = 0.75,label = 'Reynolds et al. (2014) $\delta^{18}$O')
ax21.set_ylabel('$\delta^{18}$O')

ax22 = ax21.twinx()
#l22 = ax22.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')
l22 = ax22.plot(years,mean_density2,'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland seawater density')
ax22.set_ylabel('Density anomaly\n(kg/m$^3$)')



lns = l21b+l22 
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})

###
#3rd panel down (what is driving density?)
###

ax31 = fig.add_subplot(413)
#l31 = ax31.plot(years,rm.running_mean(mean_density2,smoothing_val),'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland density - note filtering poss. messing up ends')
l31 = ax31.plot(years,mean_density2,'y',linewidth=wdth,alpha=0.7,label = 'PMIP3/CMIP5 N. Iceland density - note filtering poss. messing up ends')
#l31b = ax31.plot(years,rm.running_mean(temperature_meaned_density2,smoothing_val),'r',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to salinity)')
l31b = ax31.plot(years,temperature_meaned_density2,'r',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to salinity)')
#l31c = ax31.plot(years,rm.running_mean(salinity_meaned_density2,smoothing_val),'b',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to temperature)')
l31c = ax31.plot(years,salinity_meaned_density2,'b',linewidth=wdth/2,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland density (due to temperature)')

ax31.set_ylabel('Density anomaly\n(kg/m$^3$)')

lns = l31+l31b+l31c
 
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})

###
#4th panel down (Precipitation driving salinity?)
###

ax41 = fig.add_subplot(414)
#l41 = ax41.plot(years,rm.running_mean(mean_salinity2,smoothing_val),'b',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland salinity')
l41 = ax41.plot(years,mean_salinity2,'b',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland salinity')
ax42=ax41.twinx()
#l42 = ax42.plot(years,rm.running_mean(precipitation2,smoothing_val),'r',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland precipitation - note signal processing seems to have flipped sign of salinity')
l42 = ax42.plot(years,precipitation2,'r',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland precipitation - note signal processing seems to have flipped sign of salinity')
# l42 = ax42.plot(years,rm.running_mean(precipitation2,smoothing_val)/1.0e6,'r',linewidth=wdth,alpha=0.5,label = 'PMIP3/CMIP5 N. Iceland precipitation - note signal processing seems to have flipped sign of salinity')


ax43=ax41.twinx()
l43 = ax43.plot(voln_n[:,0],voln_n[:,1]/np.nanmax(voln_n[:,1]),'k',linewidth = wdth,alpha = 0.2,label = 'volcanic index (normalised to fill panel)')

ax41.set_ylabel('salinity anomaly')
ax42.set_ylabel('precipitation anomaly x10$^{-6}$')
ax41.set_xlabel('Calendar Year')

ax43.axis('off')

lns = l41+l42+l43
 
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':8})


ax11.set_xlim([850,1850])
ax21.set_xlim([850,1850])
ax31.set_xlim([850,1850])
ax41.set_xlim([850,1850])
#plt.show(block = True)
plt.savefig('/home/ph290/Documents/figures/palaeoamo/AMOC_and_n_aceland_density_II_hres.png')


'''

models2 = models.copy()
models2 = list(models2)
models2.remove('MRI-CGCM3')
models2 = np.array(models2)

west = -24
east = -13
south = 65
north = 67

west2 = -180
east2 = 180
south2 = 20
north2 = 90

cube = iris.load_cube(directory+models2[0]+'_sos_past1000_r1i1p1*_regridded_not_vertically.nc')
iris.coord_categorisation.add_year(cube, 'time', name='year2')
loc = np.where(cube.coord('year2').points == 1850)
loc2 = cube.coord('time').points[loc[0]]
cube2 = cube.extract(iris.Constraint(time = loc2))
cube2 = cube2.collapsed('depth',iris.analysis.MEAN)


for yr in years:
	tmp_array = np.array([np.shape(cube2)[0],np.shape(cube2)[1],np.size(models2)])
	for i,model in enumerate(models2):
		cube = iris.load_cube(directory+models2[0]+'_sos_past1000_r1i1p1*_regridded_not_vertically.nc')
		iris.coord_categorisation.add_year(cube, 'time', name='year2')
		loc = np.where(cube.coord('year2').points == yr)
		loc2 = cube.coord('time').points[loc[0]]
		cube2 = cube.extract(iris.Constraint(time = loc2))
		cube2 = cube2.collapsed('depth',iris.analysis.MEAN)
		tmp_array[:,:,i] = cube2.data
	
	
		s_cube = iris.load_cube(directory+model+'_sos_past1000_r1i1p1*_regridded_not_vertically.nc')
		coord = s_cube.coord('time')
		dt = coord.units.num2date(coord.points)
		years = np.array([coord.units.num2date(value).year for value in coord.points])
		temporary_cube = s_cube.intersection(longitude = (west, east))
		s_cube_n_iceland = temporary_cube.intersection(latitude = (south, north))
		try:
			s_cube_n_iceland.coord('latitude').guess_bounds()
			s_cube_n_iceland.coord('longitude').guess_bounds()
		except:
			print 'already have bounds'
		grid_areas = iris.analysis.cartography.area_weights(s_cube_n_iceland)
		s_cube_n_iceland_mean = s_cube_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data





##############################################
#            figure 3                        #
##############################################

###
#era interim nao precipitatoin
###

file = '/data/temp/ph290/era_interim/era_moisture_flux_ann_mean.nc'
era_cube = iris.load_cube(file)
era_cube = era_cube[:-1]
era_cube.data = scipy.signal.filtfilt(b, a, era_cube.data,axis = 0)
coord = era_cube.coord('time')
dt = coord.units.num2date(coord.points)
era_year = np.array([coord.units.num2date(value).year for value in coord.points])

nao = np.genfromtxt('/home/ph290/data0/misc_data/norm.nao.monthly.b5001.current.ascii.table',skip_footer = 1)
#http://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/JFM_season_nao_index.shtml
nao_year = nao[:,0]
jfm_nao_data = np.mean(nao[:,1:4],axis = 1)
jfm_nao_data = scipy.signal.filtfilt(b, a, jfm_nao_data)
loc = np.where(nao_year >= np.min(era_year))
nao_year = nao_year[loc]
jfm_nao_data = jfm_nao_data[loc]
jfm_nao_data = jfm_nao_data-np.mean(jfm_nao_data)

loc2_low = np.where(jfm_nao_data < 0)
loc2_high = np.where(jfm_nao_data > 0)

era_cube_low = era_cube[loc2_low].collapsed('time',iris.analysis.MEAN)
era_cube_high = era_cube[loc2_high].collapsed('time',iris.analysis.MEAN)

west = -180
east = 180
south = 20
north = 90

temporary_cube = era_cube_low.intersection(longitude = (west, east))
era_cube_low = temporary_cube.intersection(latitude = (south, north))
temporary_cube = era_cube_high.intersection(longitude = (west, east))
era_cube_high = temporary_cube.intersection(latitude = (south, north))


###
#Looking at pattern of precipitation change
###

years = np.array(years)
loc = np.where((mean_salinity2[100:-100] - np.mean(mean_salinity2[100:-100])) > 0.0)
tmp_years = years[100:-100]
high_years = tmp_years[loc[0]]

loc = np.where((precipitation2[100:-100] - np.mean(precipitation2[100:-100])) < 0.0)
tmp_years = years[100:-100]
low_years = tmp_years[loc[0]]

models = np.array(models)

pr_high = np.zeros([models.size,180,360])
pr_high[:] = np.NAN
pr_low = pr_high.copy()

for i,model in enumerate(models):
	print model
	pr_cube = iris.load_cube(directory+model+'_pr_past1000_r*_regridded.nc')
	try:
		pr_depths = pr_cube.coord('depth').points
		pr_cube = pr_cube.extract(iris.Constraint(depth = np.min(pr_depths)))
	except:
		print 'no precipitation depth coordinate'
	pr_cube.data = scipy.signal.filtfilt(b, a, pr_cube.data,axis = 0)
	# years
	coord = pr_cube.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	common_high_years = np.array(list(set(year_tmp).intersection(high_years)))
	high_yr_index = np.nonzero(np.in1d(year_tmp,common_high_years))[0]
	common_low_years = np.array(list(set(year_tmp).intersection(low_years)))
	low_yr_index = np.nonzero(np.in1d(year_tmp,common_low_years))[0]
	pr_cube_high = pr_cube[high_yr_index].collapsed('time',iris.analysis.MEAN)
	pr_cube_low = pr_cube[low_yr_index].collapsed(['time'],iris.analysis.MEAN)
	pr_high[i,:,:] = pr_cube_high.data
	pr_low[i,:,:] = pr_cube_low.data




pr_cube_high_mean = pr_cube_high.copy()
pr_cube_low_mean = pr_cube_low.copy()

pr_cube_high_mean.data = np.mean(pr_high,axis = 0)
pr_cube_low_mean.data = np.mean(pr_low,axis = 0)


west = -180
east = 180
south = 20
north = 90

temporary_cube = pr_cube_high_mean.intersection(longitude = (west, east))
pr_cube_high_mean = temporary_cube.intersection(latitude = (south, north))
temporary_cube = pr_cube_low_mean.intersection(longitude = (west, east))
pr_cube_low_mean = temporary_cube.intersection(latitude = (south, north))

###
#Looking at volcanic-driven pattern of precipitation change
###


volc_yrs = voln_n[:,0]
voln_data = voln_n[:,1]

loc = np.where((volc_yrs >= 950) & (volc_yrs < 1750))
volc_yrs = volc_yrs[loc]
voln_data = voln_data[loc]
volc_yrs = np.floor(volc_yrs)
volc_yrs2 = np.unique(volc_yrs)

voln_data2 = volc_yrs2.copy()

for i,temp_yrs in enumerate(volc_yrs2):
	loc = np.where(volc_yrs == temp_yrs)
	voln_data2[i] = np.mean(voln_data[loc])

voln_data2 = rmp.running_mean_post(voln_data2,1)

loc = np.where(voln_data2 > np.median(voln_data2))
tmp_years = years[100:-100]
high_years = tmp_years[loc[0]]

loc = np.where(voln_data2 < np.median(voln_data2))
tmp_years = years[100:-100]
low_years = tmp_years[loc[0]]

pr_high_volc = np.zeros([models.size,180,360])
pr_high_volc[:] = np.NAN
pr_low_volc = pr_high_volc.copy()

for i,model in enumerate(models):
	print model
	pr_cube = iris.load_cube(directory+model+'_pr_past1000_r*_regridded.nc')
	try:
		pr_depths = pr_cube.coord('depth').points
		pr_cube = pr_cube.extract(iris.Constraint(depth = np.min(pr_depths)))
	except:
		print 'no precipitatoin depth coordinate'
	pr_cube.data = scipy.signal.filtfilt(b, a, pr_cube.data,axis = 0)
	# years
	coord = t_cube_n_iceland.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	common_high_years = np.array(list(set(year_tmp).intersection(high_years)))
	high_yr_index = np.nonzero(np.in1d(year_tmp,common_high_years))[0]
	common_low_years = np.array(list(set(year_tmp).intersection(low_years)))
	low_yr_index = np.nonzero(np.in1d(year_tmp,common_low_years))[0]
	pr_cube_high_volc = pr_cube[high_yr_index].collapsed('time',iris.analysis.MEAN)
	pr_cube_low_volc = pr_cube[low_yr_index].collapsed(['time'],iris.analysis.MEAN)
	pr_high_volc[i,:,:] = pr_cube_high_volc.data
	pr_low_volc[i,:,:] = pr_cube_low_volc.data


pr_cube_high_volc_mean = pr_cube_high_volc.copy()
pr_cube_low_volc_mean = pr_cube_low_volc.copy()

pr_cube_high_volc_mean.data = np.mean(pr_high_volc,axis = 0)
pr_cube_low_volc_mean.data = np.mean(pr_low_volc,axis = 0)


###
#Producing plot
###


plt.close('all')
fig = plt.figure(figsize = (20,10))
ax1 = plt.subplot(131,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
change_precip = pr_cube_low_mean-pr_cube_high_mean
my_plot = iplt.contourf(change_precip,np.linspace(-15.0e-7,15.0e-7,31),cmap='bwr')
ax1.add_feature(cfeature.LAND,facecolor='#f6f6f6')
plt.gca().coastlines()
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(pr_cube_high_mean.long_name+' ('+format(pr_cube_high_mean.units)+')')
plt.title('PMIP3 high/low precip. composites\nrelating to high/low salinity years')

ax2 = plt.subplot(132,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax2.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
change_precip = pr_cube_low_volc_mean-pr_cube_high_volc_mean
my_plot = iplt.contourf(change_precip,np.linspace(-15.0e-7,15.0e-7,31),cmap='bwr')
ax2.add_feature(cfeature.LAND,facecolor='#f6f6f6')
plt.gca().coastlines()
#ax.add_feature(cfeature.RIVERS)
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(pr_cube_low_volc_mean.long_name+' ('+format(pr_cube_low_volc_mean.units)+')')
plt.title('PMIP3 precip:\nhigh/low volcanic year composites')

ax3 = plt.subplot(133,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax3.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
change_precip = era_cube_high-era_cube_low
my_plot = iplt.contourf(change_precip,np.linspace(-8.0e-6,8e-6,31),cmap='bwr')
ax3.add_feature(cfeature.LAND,facecolor='#f6f6f6')
plt.gca().coastlines()
#ax.add_feature(cfeature.RIVERS)
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label(era_cube_high.long_name+' ('+format(era_cube_high.units)+')')
plt.title('ERA interim precip:\nhigh/low NAO composites')

plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/palaeoamo/precip_composites_II_hres.png')

'''


def multi_model_mean(var):
	cube = iris.load_cube(directory+'HadCM3'+'_'+var+'_past1000_r1i1p1*.nc')
	try:
		iris.coord_categorisation.add_year(cube, 'time', name='year2')
	except:
		print 'year2 already exists'
	loc = np.where(cube.coord('year2').points == 1850)
	loc2 = cube.coord('time').points[loc[0]]
	cube2 = cube.extract(iris.Constraint(time = loc2))
	try:
		cube2 = cube2.collapsed('depth',iris.analysis.MEAN)
	except:
		print 'only one depth'
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	years2 = np.array([coord.units.num2date(value).year for value in coord.points])
	multimodel_mean_cube_data = cube.data
	for i,model in enumerate(models2):
		cube = iris.load_cube(directory+model+'_'+var+'_past1000_r1i1p1*.nc')
		cube.data = scipy.signal.filtfilt(b, a, cube.data,axis = 0)
		iris.fileformats.netcdf.save(cube, '/home/ph290/data0/tmp/'+model+'.nc', netcdf_format='NETCDF4')
	for j,yr in enumerate(years2):
		print yr
		tmp_array = np.zeros([np.shape(cube2)[0],np.shape(cube2)[1],np.size(models2)])
		tmp_array[:] = np.nan
		for i,model in enumerate(models2):
			cube = iris.load_cube('/home/ph290/data0/tmp/'+model+'.nc')
			try:
				iris.coord_categorisation.add_year(cube, 'time', name='year2')
			except:
				print 'year2 already exists'
			loc = np.where(cube.coord('year2').points == yr)
			if np.size(loc) != 0:
				loc2 = cube.coord('time').points[loc[0]]
				cube2 = cube.extract(iris.Constraint(time = loc2))
				try:
					cube2 = cube2.collapsed('depth',iris.analysis.MEAN)
				except:
					print 'only one depth'
				tmp_array[:,:,i] = cube2.data
		meaned_year = scipy.stats.nanmean(tmp_array,axis = 2)
		multimodel_mean_cube_data[j,:,:] = meaned_year
	return multimodel_mean_cube_data
		

var ='sos'
cube = iris.load_cube(directory+'HadCM3'+'_'+var+'_past1000_r1i1p1*.nc')
sos_cube_mm_mean = cube.copy()
sos_cube_mm_mean.data = multi_model_mean(var)

var ='pr'
cube = iris.load_cube(directory+'HadCM3'+'_'+var+'_past1000_r1i1p1*.nc')
pr_cube_mm_mean = cube.copy()
pr_cube_mm_mean.data = multi_model_mean(var)

var ='tos'
cube = iris.load_cube(directory+'HadCM3'+'_'+var+'_past1000_r1i1p1*.nc')
tos_cube_mm_mean = cube.copy()
tos_cube_mm_mean.data = multi_model_mean(var)

#ADD DENSITY

cube = iris.load_cube(directory+'HadCM3'+'_'+var+'_past1000_r1i1p1*.nc')
try:
	iris.coord_categorisation.add_year(cube, 'time', name='year2')
except:
	print 'year2 already exists'



loc = np.where(cube.coord('year2').points == 1850)
loc2 = cube.coord('time').points[loc[0]]
cube2 = cube.extract(iris.Constraint(time = loc2))
try:
	cube2 = cube2.collapsed('depth',iris.analysis.MEAN)
except:
	print 'only one depth'



coord = cube.coord('time')
dt = coord.units.num2date(coord.points)
years2 = np.array([coord.units.num2date(value).year for value in coord.points])
multimodel_mean_cube_data_density = cube.data
multimodel_mean_cube_data_density_t_const = multimodel_mean_cube_data_density.copy()
multimodel_mean_cube_data_density_s_const = multimodel_mean_cube_data_density.copy()

time_meaned_data = {}

for i,model in enumerate(models2):
	time_meaned_data[model] = {}
	cube = iris.load_cube(directory+model+'_sos_past1000_r1i1p1*.nc')
	mask = cube.data.mask
	cube_mean = cube.collapsed('time',iris.analysis.MEAN)
	cube.data = scipy.signal.filtfilt(b, a, cube.data,axis = 0)
	cube.data = ma.masked_array(cube.data)
	cube.data.mask = mask
	cube += cube_mean
	iris.fileformats.netcdf.save(cube, '/home/ph290/data0/tmp/'+model+'_s.nc', netcdf_format='NETCDF4')
	time_meaned_data[model]['sos'] = cube_mean
	cube = iris.load_cube(directory+model+'_tos_past1000_r1i1p1*.nc')
	mask = cube.data.mask
	cube_mean = cube.collapsed('time',iris.analysis.MEAN)
	cube.data = scipy.signal.filtfilt(b, a, cube.data,axis = 0)
	cube.data = ma.masked_array(cube.data)
	cube.data.mask = mask
	cube += cube_mean
	iris.fileformats.netcdf.save(cube, '/home/ph290/data0/tmp/'+model+'_t.nc', netcdf_format='NETCDF4')
	time_meaned_data[model]['tos'] = cube_mean


for j,yr in enumerate(years2):
	print yr
	tmp_array_s = np.zeros([np.shape(cube2)[0],np.shape(cube2)[1],np.size(models2)])
	tmp_array_s[:] = np.nan
	for i,model in enumerate(models2):
		cube_s = iris.load_cube('/home/ph290/data0/tmp/'+model+'_s.nc')
		try:
			iris.coord_categorisation.add_year(cube_s, 'time', name='year2')
		except:
			print 'year2 already exists'
		loc = np.where(cube_s.coord('year2').points == yr)
		if np.size(loc) != 0:
			loc2 = cube_s.coord('time').points[loc[0]]
			cube2_s = cube_s.extract(iris.Constraint(time = loc2))
			try:
				cube2_s = cube2_s.collapsed('depth',iris.analysis.MEAN)
			except:
				print 'only one depth'
			tmp_array_s[:,:,i] = cube2_s.data
	tmp_array_t = np.zeros([np.shape(cube2)[0],np.shape(cube2)[1],np.size(models2)])
	tmp_array_t[:] = np.nan
	for i,model in enumerate(models2):
		cube_t = iris.load_cube('/home/ph290/data0/tmp/'+model+'_t.nc')
		try:
			iris.coord_categorisation.add_year(cube_t, 'time', name='year2')
		except:
			print 'year2 already exists'
		loc = np.where(cube_t.coord('year2').points == yr)
		if np.size(loc) != 0:
			loc2 = cube_t.coord('time').points[loc[0]]
			cube2_t = cube_t.extract(iris.Constraint(time = loc2))
			try:
				cube2_t = cube2_t.collapsed('depth',iris.analysis.MEAN)
			except:
				print 'only one depth'
			tmp_array_t[:,:,i] = cube2_t.data
	density = seawater.dens(tmp_array_s,tmp_array_t-273.15)
	meaned_year = scipy.stats.nanmean(density,axis = 2)
	multimodel_mean_cube_data_density[j,:,:] = meaned_year
	#tos const
	tmp_array_t_const = tmp_array_t.copy()
	for i,model in enumerate(models2):
		tmp_array_t_const[:,:,i] = time_meaned_data[model]['tos'].data
	density = seawater.dens(tmp_array_s,tmp_array_t_const-273.15)
	meaned_year = scipy.stats.nanmean(density,axis = 2)
	multimodel_mean_cube_data_density_t_const[j,:,:] = meaned_year
	#sos const
	tmp_array_s_const = tmp_array_s.copy()
	for i,model in enumerate(models2):
		tmp_array_s_const[:,:,i] = time_meaned_data[model]['sos'].data
	density = seawater.dens(tmp_array_s_const,tmp_array_t_const-273.15)
	meaned_year = scipy.stats.nanmean(density,axis = 2)
	multimodel_mean_cube_data_density_s_const[j,:,:] = meaned_year


mmm_density_cube = cube.copy()
mmm_density_cube.data = multimodel_mean_cube_data_density

mmm_density_cube_s_const = cube.copy()
mmm_density_cube_s_const.data = multimodel_mean_cube_data_density_s_const

mmm_density_cube_t_const = cube.copy()
mmm_density_cube_t_const.data = multimodel_mean_cube_data_density_t_const

# 	tmp_density = seawater.dens(s_cube_n_iceland_mean,t_cube_n_iceland_mean-273.15)
# 	density_data[model] = {}
# 	density_data[model]['temperature'] = t_cube_n_iceland_mean
# 	density_data[model]['salinity'] = s_cube_n_iceland_mean
# 	density_data[model]['density'] = tmp_density
# 	density_data[model]['temperature_meaned_density'] = tmp_temp_mean_density
# 	density_data[model]['salinity_meaned_density'] = tmp_sal_mean_density
# 	density_data[model]['precipitation'] = pr_cube_n_iceland_mean
# 	density_data[model]['years'] = year_tmp


loc = np.where((mean_salinity2[100:-100] - np.mean(mean_salinity2[100:-100])) > 0.0)
tmp_years = years2[100:-100]
high_years = tmp_years[loc[0]]

loc = np.where((precipitation2[100:-100] - np.mean(precipitation2[100:-100])) < 0.0)
tmp_years = years2[100:-100]
low_years = tmp_years[loc[0]]

common_low_years = np.array(list(set(years2).intersection(low_years)))
low_yr_index = np.nonzero(np.in1d(years2,common_low_years))[0]
low_so_cube = so_cube_mm_mean[low_yr_index].collapsed(['time'],iris.analysis.MEAN)
low_so_cube.data = ma.masked_invalid(low_so_cube.data)

common_high_years = np.array(list(set(years2).intersection(high_years)))
high_yr_index = np.nonzero(np.in1d(years2,common_high_years))[0]
high_so_cube = so_cube_mm_mean[high_yr_index].collapsed(['time'],iris.analysis.MEAN)

low_pr_cube = pr_cube_mm_mean[low_yr_index].collapsed(['time'],iris.analysis.MEAN)
low_pr_cube.data = ma.masked_invalid(low_pr_cube.data)

high_yr_index = np.nonzero(np.in1d(years2,common_high_years))[0]
high_pr_cube = pr_cube_mm_mean[high_yr_index].collapsed(['time'],iris.analysis.MEAN)


var = 'pr'

west = -24
east = -13
south = 65
north = 67

multi_model_dataset = np.zeros([np.size(years),np.size(models2)])
multi_model_dataset[:] = np.nan

for i,model in enumerate(models2):
	cube = iris.load_cube(directory+model+'_'+var+'_past1000_r1i1p1*.nc')
	try:
		iris.coord_categorisation.add_year(cube, 'time', name='year2')
	except:
		print 'year2 already exists'
	cube = cube.intersection(longitude = (west, east))
	cube = cube.intersection(latitude = (south, north))
	cube.data = scipy.signal.filtfilt(b, a, cube.data,axis = 0)
	cube.coord('latitude').guess_bounds()
	cube.coord('longitude').guess_bounds()
	grid_areas = iris.analysis.cartography.area_weights(cube)
	ts = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas).data
# 	ts = scipy.signal.filtfilt(b, a, ts.data,axis = 0)
	ts = ts/(np.max(ts)-np.min(ts))
 	ts = ts -np.mean(ts)
	yrs_tmp = cube.coord('year2').points
	for j,yr in enumerate(years):
		loc = np.where(yrs_tmp == yr)
		if np.size(loc) != 0:
			multi_model_dataset[j,i] = ts[loc]
			

multi_model_mean = np.mean(multi_model_dataset,axis = 1)
plt.plot(multi_model_mean)
plt.show()


plt.close('all')
fig = plt.figure(figsize = (20,10))
ax1 = plt.subplot(121,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
change_sos = high_so_cube-low_so_cube
my_plot = iplt.contourf(change_sos,np.linspace(-0.2,0.2,31),cmap='bwr')
ax1.add_feature(cfeature.LAND,facecolor='#f6f6f6')
plt.gca().coastlines()
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label('anomaly (psu)')
plt.title('Salinity: PMIP3 multi model mean high minus low N. Iceland salinity years')

ax2 = plt.subplot(122,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
ax2.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
change_precip = high_pr_cube-low_pr_cube
my_plot = iplt.contourf(change_precip,np.linspace(-1.5e-6,1.5e-6,31),cmap='bwr')
ax2.add_feature(cfeature.LAND,facecolor='#f6f6f6')
plt.gca().coastlines()
#ax.add_feature(cfeature.RIVERS)
bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
bar.set_label('anomaly (kg m-2 s-1)')
plt.title('Precipitation: PMIP3 multi model mean high minus low N. Iceland salinity years')

plt.show()



west = -24
east = -13
south = 65
north = 67



temporary_cube = pr_cube_mm_mean.intersection(longitude = (west, east))
pr_cube_mm_mean_reg = temporary_cube.intersection(latitude = (south, north))

pr_cube_mm_mean_reg.coord('latitude').guess_bounds()
pr_cube_mm_mean_reg.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(pr_cube_mm_mean_reg)
pr_ts = pr_cube_mm_mean_reg.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
try:
	iris.coord_categorisation.add_year(pr_ts, 'time', name='year2')
except:
	print 'year2 already exists'
	

y = pr_ts.data
y = y/(np.max(y)-np.min(y))
y = y -np.mean(y)

temporary_cube = sos_cube_mm_mean.intersection(longitude = (west, east))
sos_cube_mm_mean_reg = temporary_cube.intersection(latitude = (south, north))

sos_cube_mm_mean_reg.coord('latitude').guess_bounds()
sos_cube_mm_mean_reg.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(sos_cube_mm_mean_reg)
sos_ts = sos_cube_mm_mean_reg.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
try:
	iris.coord_categorisation.add_year(sos_ts, 'time', name='year2')
except:
	print 'year2 already exists'
	

y1 = sos_ts.data
y1 = y1/(np.max(y1)-np.min(y1))
y1 = y1 -np.mean(y1)


west = -24
east = -13
south = 65
north = 72


temporary_cube = mmm_density_cube.intersection(longitude = (west, east))
density_cube_mm_mean_reg = temporary_cube.intersection(latitude = (south, north))
density_cube_mm_mean_reg.data = ma.masked_invalid(density_cube_mm_mean_reg.data)

density_cube_mm_mean_reg.coord('latitude').guess_bounds()
density_cube_mm_mean_reg.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(density_cube_mm_mean_reg)
density_ts = density_cube_mm_mean_reg.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
try:
	iris.coord_categorisation.add_year(density_ts, 'time', name='year2')
except:
	print 'year2 already exists'
	

y2 = density_ts.data
y2 = y2/(ma.max(y2)-ma.min(y2))
y2 = y2 - ma.mean(y2)

# y2 = mean_density2
# y2 = y2/(np.max(y2)-np.min(y2))
# y2 = y2 -np.mean(y2)

plt.close('all')
fig = plt.figure(figsize=(12,8),dpi=60)
ax11 = fig.add_subplot(111)
l1 = ax11.plot(pr_ts.coord('year2').points,rm.running_mean(y*(-1.0),10),'r')
ax12 = ax11.twinx()
l2 = ax12.plot(sos_ts.coord('year2').points,rm.running_mean(y1,10),'b')
ax13 = ax12.twinx()
l3 = ax13.plot(years,rm.running_mean(y2,10),'g')
ax11.set_ylim(-0.2,0.2)
ax12.set_ylim(-0.5,0.5)
plt.show()

#REVISIT DENSITY - calculate all sensibly as before, and make sure precip is right way to do what it should


