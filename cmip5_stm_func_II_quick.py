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

'''

'''
#producing an Atlantic mask (mask masked and Atlantic has value of 1, elsewhere zero)
'''

input_file = '/media/usb_external1/cmip5/last1000/CCSM4_vo_past1000_regridded.nc'
cube = iris.load_cube(input_file)

tmp_cube = cube[0,16].copy()
tmp_cube = tmp_cube*0.0

location = -30

for y in np.arange(180):
    flag = 0
    tmp = tmp_cube.data.mask[y,:]
    tmp2 = tmp_cube.data[y,:]
    for x in np.arange(360):
        if tmp[location] == True:
            flag = 1
        if ((tmp[location] == False) & (flag == 0)):
            tmp2[location] = 1
        tmp = np.roll(tmp,+1)
        tmp2 = np.roll(tmp2,+1)
    tmp_cube.data.mask[y,:] = tmp
    tmp_cube.data.data[y,:] = tmp2.data

location = location+1

for y in np.arange(180):
    flag = 0
    tmp = tmp_cube.data.mask[y,:]
    tmp2 = tmp_cube.data[y,:]
    for x in np.arange(360):
        if tmp[location] == True:
            flag = 1
        if ((tmp[location] == False) & (flag == 0)):
            tmp2[location] = 1
        tmp = np.roll(tmp,-1)
        tmp2 = np.roll(tmp2,-1)
    tmp_cube.data.mask[y,:] = tmp
    tmp_cube.data.data[y,:] = tmp2.data

tmp_cube.data.data[150:180,:] = 0.0
tmp_cube.data.data[0:40,:] = 0.0
tmp_cube.data.data[:,20:180] = 0.0
tmp_cube.data.data[:,180:280] = 0.0

loc = np.where(tmp_cube.data.data == 0.0)
tmp_cube.data.mask[loc] = True

mask1 = tmp_cube.data.mask


# plt.close('all')
# qplt.contourf(cube[0,28])
# plt.show(block = False)

# my_dir = '/media/usb_external1/cmip5/msftmyz/last1000/'
# files = np.array(glob.glob(my_dir+'/*'+'MPI-ESM-P'+'_*msftmy*.nc'))
# cube2 = iris.load_cube(files)[:,0,:,:]
# loc = np.where((cube2.coord('grid_latitude').points >= 30) & (cube2.coord('grid_latitude').points <= 50))
# lat = cube2.coord('grid_latitude').points[loc]
# sub_cube = cube2.extract(iris.Constraint(grid_latitude = lat))
# stream_function_tmp = sub_cube.collapsed(['depth','grid_latitude'],iris.analysis.MAX)
# coord = stream_function_tmp.coord('time')
# dt = coord.units.num2date(coord.points)
# year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
# tmp = stream_function_tmp.data/1.0e9
# mpi_official_strmfun_26 = (tmp[np.logical_not(np.isnan(tmp))])
# mpi_official_year = (year_tmp[np.logical_not(np.isnan(tmp))])

# f0 = plt.figure()
# qplt.contourf(cube2[0],51)
# plt.show(block = False)


'''
#calculating stream function
'''

files = glob.glob('/media/usb_external1/cmip5/last1000_vo_amoc/*_vo_*.nc')

models = []
max_strm_fun = []
max_strm_fun_26 = []
max_strm_fun_45 = []
model_years = []

for file in files:

    model = file.split('/')[5].split('_')[0]
    print model
    models.append(model)
    cube = iris.load_cube(file)

    print 'applying mask'

    try:
            levels =  np.arange(cube.coord('depth').points.size)
    except:
            levels = np.arange(cube.coord('ocean sigma over z coordinate').points.size)

    for level in levels:
        print 'level: '+str(level)
        for year in np.arange(cube.coord('time').points.size):
            #print 'year: '+str(year)
            tmp = cube.lazy_data()
            mask2 = tmp[year,level,:,:].masked_array().mask
            tmp_mask = np.ma.mask_or(mask1, mask2)
            tmp[year,level,:,:].masked_array().mask = tmp_mask


    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube[0])
    grid_areas = np.sqrt(grid_areas)

    shape = np.shape(cube)
    tmp = cube[0].collapsed('longitude',iris.analysis.SUM)
    collapsed_data = np.tile(tmp.data,[shape[0],1,1])

    print 'collapsing cube along longitude'
    try:
            slices = cube.slices(['depth', 'latitude','longitude'])
    except:
            slices = cube.slices(['ocean sigma over z coordinate', 'latitude','longitude'])
    for i,t_slice in enumerate(slices):
            #print 'year:'+str(i)
            t_slice *= grid_areas
            collapsed_data[i] = t_slice.collapsed('longitude',iris.analysis.SUM).data

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

    thickness = np.flipud(np.rot90(np.tile(thickness,[180,1])))

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

# plt.close('all')
# # plt.plot(tmp_strm_fun/(110*110))
# plt.plot(tmp_strm_fun)
# plt.plot(mpi_official_strmfun_26)
# plt.show(block = False)

# f2 = plt.figure()		
# CS = plt.contourf(np.arange(180)-90,depths,collapsed_data[0],51)
# plt.colorbar(CS)
# plt.show(block = False)

# f3 = plt.figure()
# CS = plt.contourf(np.arange(180)-90,depths,np.fliplr(tmp*-1.0*1.0e-3),51)
# plt.colorbar(CS)
# plt.show(block = False)


for i,model in enumerate(models):
	plt.plot(model_years[i],max_strm_fun[i])


plt.show(block = False)

###
#read in temperature
###



amo_box_tas = []
model_years_tas = []

for i,model in enumerate(models):
	print 'processing: '+model
	file = glob.glob('/media/usb_external1/cmip5/tas_regridded/'+model+'_tas_past1000_regridded.nc')
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
loc = np.where(amo_yr <= 1850)
amo_yr = amo_yr[loc]
amo_data = amo_data[loc]
amo_data = signal.detrend(amo_data)

'''
#read in volc data
'''

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
#plotting
###


linestyles = ['-',':','-.']

smoothing_val = 5

plt.close('all')
fig = plt.figure(figsize = (6,20))

for i,model in enumerate(models):
        ax1 = fig.add_subplot(np.size(models),1,i+1)
        tmp = amo_box_tas[i].data
	tmp = rm.running_mean(tmp,smoothing_val)
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years_tas[i] <= 1850))
        tmp = tmp[loc]
        #ax1.plot(model_years_tas[i][loc],signal.detrend(tmp),'r',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
        ax1.plot(model_years_tas[i][loc],tmp-np.mean(tmp),'r',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
        loc = np.where((amo_yr >= 850) & (amo_yr <= 1850))
        ax1.plot(amo_yr[loc],amo_data[loc] - np.mean(amo_data[loc]),'k',linewidth=2,alpha = 0.5)
        ax2 = ax1.twinx()
        tmp = max_strm_fun_26[i]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years[i] <= 1850))
        tmp = tmp[loc]
        #ax2.plot(model_years[i][loc],signal.detrend(tmp),'b',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
        ax2.plot(model_years[i][loc],tmp,'b',linewidth=2,alpha = 0.5,linestyle = linestyles[0])
        ax3 = ax2.twinx()
        ax3.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=2,alpha = 0.2)
	ax3.plot(voln_s[:,0],voln_s[:,1],'b',linewidth=2,alpha = 0.2)
        ax3.set_ylim([0,0.8])
        ax1.set_xlim([800,1850])
	ax1.set_title(model)
	

plt.tight_layout()
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/amo_fig2.png')


#AVERAGE TOGETHER!

smoothing_val = 5

all_years = np.linspace(850,1850,(1851-850))
average_tas = np.empty([np.size(all_years),np.size(models)-5])
average_tas[:] = np.NAN
average_str = np.empty([np.size(all_years),np.size(models)-5])
average_str[:] = np.NAN

counter = -1
for i,model in enumerate(models):
    if model not in ['FGOALS-gl','GISS-E2-R','CSIRO-Mk3L-1-2']:
        counter += 1
        tmp = amo_box_tas[i].data
        tmp = rm.running_mean(tmp,smoothing_val)
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years_tas[i] <= 1850) & (model_years_tas[i] >= 850))
        tmp = tmp[loc]
        yrs = model_years_tas[i][loc]
        data = signal.detrend(tmp)
        for j,yr in enumerate(all_years):
                try:
                        loc2 = np.where(yrs == yr)
                        average_tas[j,counter] = data[loc2]
                except:
                        None
        tmp = max_strm_fun_26[i]
        tmp = rm.running_mean(tmp,smoothing_val)
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years[i] <= 1850) & (model_years[i] >= 850))
        tmp = tmp[loc]
        yrs = model_years[i][loc]
        data = signal.detrend(tmp)
        data = data/np.mean(data)
        for j,yr in enumerate(all_years):
                try:
                        loc2 = np.where(yrs == yr)
                        average_str[j,counter] = data[loc2]
                except:
                        None
        

	
average_tas2 = np.mean(average_tas,axis = 1)
average_str2 = np.mean(average_str,axis = 1)

#wan_data = np.genfromtxt('/home/ph290/data0/misc_data/wanamaker_data.csv',skip_header = 1,delimiter = ',')



plt.close('all')
fig = plt.figure(figsize = (12,5))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(all_years,average_tas2,'r',linewidth=3,alpha = 0.5,linestyle = linestyles[0])
ax1.plot(amo_yr,signal.detrend(amo_data),'k',linewidth=3,alpha = 0.5)
ax2 = ax1.twinx()
ax2.plot(all_years,average_str2,'b',linewidth=3,alpha = 0.5,linestyle = linestyles[0])
ax3 = ax2.twinx()
ax3.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=3,alpha = 0.2)
ax3.plot(voln_s[:,0],voln_s[:,1],'b',linewidth=2,alpha = 0.2)
ax3.set_ylim([0,0.8])
#ax4 = ax3.twinx()
#ax4.scatter(wan_data[:,0],wan_data[:,1]*-1.0,color = 'red')
ax1.set_xlim([850,1850])
ax1.set_ylim(-0.5,0.5)

plt.tight_layout()
plt.show(block = False)
#plt.savefig('/home/ph290/Documents/figures/amo_fig3.png')


'''
#low-pass
'''


file1 = '/home/ph290/data0/misc_data/last_millenium_solar/tsi_SBF_11yr.txt'
data1 = np.genfromtxt(file1,skip_header = 4)
solar_year = data1[:,0]
solar_data = data1[:,1]



N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
low_cutoff=150.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
high_cutoff=150.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
middle_cuttoff_low=1.0
################
middle_cuttoff_high=100.0
################

Wn_low=timestep_between_values/low_cutoff
Wn_high=timestep_between_values/high_cutoff
Wn_mid_low=timestep_between_values/middle_cuttoff_low
Wn_mid_high=timestep_between_values/middle_cuttoff_high

b, a = scipy.signal.butter(N, Wn_low, btype='low')
b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')


all_years = np.linspace(850,1850,(1851-850))
average_tas = np.empty([np.size(all_years),np.size(models)-1])
average_tas[:] = np.NAN
average_str = np.empty([np.size(all_years),np.size(models)-1])
average_str[:] = np.NAN

counter = -1
for i,model in enumerate(models):
    if model not in ['FGOALS-gl']:
#,'GISS-E2-R']:
        counter += 1
        tmp = amo_box_tas[i].data
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years_tas[i] <= 1850) & (model_years_tas[i] >= 850))
        tmp = tmp[loc]
        data = scipy.signal.filtfilt(b2, a2, tmp)
        yrs = model_years_tas[i][loc]
        for j,yr in enumerate(all_years):
                try:
                        loc2 = np.where(yrs == yr)
                        average_tas[j,counter] = data[loc2]
                except:
                        None
        tmp = max_strm_fun_26[i]
        loc = np.where((np.logical_not(np.isnan(tmp))) & (model_years[i] <= 1850) & (model_years[i] >= 850))
        tmp = tmp[loc]
        data = scipy.signal.filtfilt(b2, a2, tmp)
        yrs = model_years[i][loc]
        #data = data/np.mean(data)
        for j,yr in enumerate(all_years):
                try:
                        loc2 = np.where(yrs == yr)
                        average_str[j,counter] = data[loc2]
                except:
                        None
        

	
average_tas2 = np.mean(average_tas,axis = 1)
average_str2 = np.mean(average_str,axis = 1)

smoothing_val = 10

y1 = rm.running_mean(average_tas2,smoothing_val)
y2 = amo_data.copy()
y2 = scipy.signal.filtfilt(b2, a2, y2)
y3 = rm.running_mean(average_str2,smoothing_val)

plt.close('all')
fig = plt.figure(figsize = (12,12))
ax1 = fig.add_subplot(3,1,1)
ax1.plot(amo_yr,y2,'k',linewidth=3,alpha = 0.8)
ax1.plot(all_years,y1,'r',linewidth=3,alpha = 0.8,linestyle = linestyles[0])
#ax2 = ax1.twinx()
#ax2.plot(all_years,y3,'b',linewidth=3,alpha = 0.2,linestyle = linestyles[0])
ax2 = ax1.twinx()
ax2.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=3,alpha = 0.2)
ax2.plot(voln_s[:,0],voln_s[:,1],'b',linewidth=2,alpha = 0.2)
ax2.set_ylim([0,0.8])
ax1.set_xlim([850,1850])
ax1.set_ylim(-0.5,0.5)

ax3 = fig.add_subplot(3,1,2)
ax3.plot(all_years,y3,'b',linewidth=3,alpha = 0.8,linestyle = linestyles[0])
ax4 = ax3.twinx()
ax4.plot(all_years,y1,'r',linewidth=3,alpha = 0.8,linestyle = linestyles[0])
ax4b = ax4.twinx()
ax4b.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=3,alpha = 0.2)
ax4b.plot(voln_s[:,0],voln_s[:,1],'b',linewidth=2,alpha = 0.2)
ax4.set_xlim([850,1850])
ax4.set_ylim(-0.5,0.5)
ax3.set_ylim(-1.5e7,1.5e7)

ax5 = fig.add_subplot(3,1,3)
ax5.plot(all_years,y3,'b',linewidth=3,alpha = 0.8,linestyle = linestyles[0])
ax6 = ax5.twinx()
ax6.plot(amo_yr,y2,'k',linewidth=3,alpha = 0.8)
ax6b = ax6.twinx()
#y4 = solar_data
#y4 = scipy.signal.filtfilt(b2, a2, y4)
#y4 = rm.running_mean(y4,smoothing_val)
#ax6b.plot(solar_year,y4,'g',linewidth=3,alpha = 0.3)
ax6b.plot(voln_n[:,0],voln_n[:,1],'k',linewidth=3,alpha = 0.2)
ax6b.plot(voln_s[:,0],voln_s[:,1],'b',linewidth=2,alpha = 0.2)
ax4.set_xlim([850,1850])
ax5.set_ylim(-1.5e7,1.5e7)

ax6.set_xlim([850,1850])
ax6.set_ylim(-0.5,0.5)
#ax4 = ax3.twinx()
#ax4.scatter(wan_data[:,0],wan_data[:,1]*-1.0,color = 'red')
plt.tight_layout()
plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/amo_fig4.png')


'''
#Can we build a simple statiustical model to help us understand the palaeo AMO?
#using volcanoes (both hemisphered), post-event smoothing and maybe the modeled AMOC?
'''

import statsmodels.api as sm
ens_size = 100

y2 = amo_data.copy()
#y2 = scipy.signal.filtfilt(b2, a2, y2)

y4 = solar_data
#y4 = scipy.signal.filtfilt(b2, a2, y4)
#y4 = rm.running_mean(y4,smoothing_val)
#solar_year
#y4

p1 = np.random.randint(1,30,ens_size)
p2 = np.random.randint(1,30,ens_size)
p3 = np.random.randint(1,30,ens_size)
#p2 = np.random.randint(-100,100,ens_size)/1000.0

start_year = 1400
end_year = 1850

mann_amo = np.interp(voln_n[(start_year-800)*36:(end_year - 800)*36,0],amo_yr,y2)
y4b = np.interp(voln_n[(start_year-800)*36:(end_year - 800)*36,0],solar_year,y4)
y4b[np.where(np.isnan(y4b))] = 0.0
amoc = np.interp(voln_n[(start_year-800)*36:(end_year - 800)*36,0],all_years,y3)
amoc[np.where(np.isnan(amoc))] = 0.0

r_squareds = np.zeros(ens_size)


for i in np.arange(ens_size):
    print i
    smth = p1[i]
    smthb = p2[i]
    smthc = p3[i]
    vns = running_mean_post.running_mean_post(voln_n[(start_year-800)*36:(end_year - 800)*36,1],smth*36.0)
    vss = rm.running_mean(voln_s[(start_year-800)*36:(end_year - 800)*36,1],smth*36.0)
    vns[np.where(np.isnan(vns))] = 0.0
    vss[np.where(np.isnan(vss))] = 0.0
    amocb = running_mean_post.running_mean_post(amoc*36,smthb*36.0)
    y = mann_amo
    x1 = vns
    x2 = vss
    x3 = running_mean_post.running_mean_post(y4b,smthc*36.0)
    x3[np.where(np.isnan(x3))] = 0.0
    x4 = amocb
    x = np.column_stack((x1,x2,x3,x4))
    #stack explanatory variables into an array
    x = sm.add_constant(x)
    #add constant to first column for some reasons
    model = sm.OLS(y,x)
    results = model.fit()
    r_squareds[i] = results.rsquared



loc = np.where(r_squareds == np.max(r_squareds))
i = loc[0][0]
smth = p1[i]
smthb = p2[i]
vns = running_mean_post.running_mean_post(voln_n[(start_year-800)*36:(end_year - 800)*36,1],smth*36.0)
vss = running_mean_post.running_mean_post(voln_s[(start_year-800)*36:(end_year - 800)*36,1],smth*36.0)
amocb = running_mean_post.running_mean_post(amoc*36,smthb*36.0)
x1 = vns
x2 = vss
x3 = y4b
x4 = amocb
y = mann_amo
x = np.column_stack((x1,x2,x3,x4))
#stack explanatory variables into an array
x = sm.add_constant(x)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

plt.close('all')
plt.plot(voln_n[(start_year-800)*36:(end_year - 800)*36,0],y)
plt.plot(voln_n[(start_year-800)*36:(end_year - 800)*36,0],results.params[4]*x4+results.params[3]*x3+results.params[2]*x2+results.params[1]*x1+results.params[0])
plt.savefig('/home/ph290/Documents/figures/test1.png')
#plt.show(block = False)


################
#just volvanoes
################


start_year = 850
end_year = 1850

mann_amo = np.interp(voln_n[(start_year-800)*36:(end_year - 800)*36,0],amo_yr,y2)

r_squareds = np.zeros(ens_size)

smth = 7

vns = running_mean_post.running_mean_post(voln_n[(start_year-800)*36:(end_year - 800)*36,1],smth*36.0)
vss = running_mean_post.running_mean_post(voln_s[(start_year-800)*36:(end_year - 800)*36,1],smth*36.0)

ens_size=1000
p1 = np.random.randint(1,8000,ens_size)/1000.0
p2 = np.random.randint(1,np.pi,ens_size)/1000.0
p3 = np.random.randint(1,8000,ens_size)/1000.0
p4 = np.random.randint(1,np.pi,ens_size)/1000.0

for i in np.arange(ens_size):
    N = np.size(vns)
    t = np.linspace(0+p2[i], p1[i]*np.pi+p2[i], N)
    data = 3.0*np.sin(t+0.001)
    t = np.linspace(0+p3[i], p1[i]*np.pi+p4[i], N)
    data2 = 3.0*np.sin(t+0.001)
    x1 = vns
    x2 = vss
    x3 = data
    x4 = data2
    y = mann_amo
    x = np.column_stack((x1,x2,x3,x4))
    #stack explanatory variables into an array
    x = sm.add_constant(x)
    #add constant to first column for some reasons
    model = sm.OLS(y,x)
    results = model.fit()
    r_squareds[i] = results.rsquared



loc = np.where(r_squareds == np.max(r_squareds))
i = loc[0][0]
t = np.linspace(0+p2[i], p1[i]*np.pi+p2[i], N)
data = 3.0*np.sin(t+0.001)
t = np.linspace(0+p3[i], p1[i]*np.pi+p4[i], N)
data2 = 3.0*np.sin(t+0.001)
x1 = vns
x2 = vss
x3 = data
x4 = data2
y = mann_amo
x = np.column_stack((x1,x2,x3,x4))
#stack explanatory variables into an array
x = sm.add_constant(x)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

plt.close('all')
plt.plot(voln_n[(start_year-800)*36:(end_year - 800)*36,0],y)
plt.plot(voln_n[(start_year-800)*36:(end_year - 800)*36,0],results.params[4]*x4+results.params[3]*x3+results.params[2]*x2+results.params[1]*x1+results.params[0])
plt.show(block = False)


############
#Read in models that have submitted strm fun diagnostics
############

my_dir = '/media/usb_external1/cmip5/msftmyz/last1000/'
files1 = np.array(glob.glob(my_dir+'*msftmyz*.nc'))


models1 = []
for file in files1:
        models1.append(file.split('/')[-1].split('_')[0])

models_unique = np.unique(np.array(models1))

cmip5_max_strmfun_26 = []
cmip5_max_strmfun_45 = []
cmip5_amo_box_tas = []
cmip5_year = []
cmip5_year2 = []

models_unique = models_unique.tolist()
#models_unique.remove('MRI-CGCM3')

for model in models_unique:
	print model
	files = np.array(glob.glob(my_dir+'/*'+model+'_*msftmy*.nc'))
	cube = iris.load_cube(files)[:,0,:,:]
	try:
		loc = np.where(cube.coord('grid_latitude').points >= 26)[0]
		lat = cube.coord('grid_latitude').points[loc[0]]
		sub_cube = cube.extract(iris.Constraint(grid_latitude = lat))
	except:
		loc = np.where(cube.coord('latitude').points >= 26)[0]
		lat = cube.coord('latitude').points[loc[0]]	
		sub_cube = cube.extract(iris.Constraint(latitude = lat))
	stream_function_tmp = sub_cube.collapsed('depth',iris.analysis.MAX)
	coord = stream_function_tmp.coord('time')
	dt = coord.units.num2date(coord.points)
	year_tmp = np.array([coord.units.num2date(value).year for value in coord.points])
	tmp = stream_function_tmp.data/1.0e9
	cmip5_max_strmfun_26.append(tmp[np.logical_not(np.isnan(tmp))])
	cmip5_year.append(year_tmp[np.logical_not(np.isnan(tmp))])
	file = glob.glob('/media/usb_external1/cmip5/tas_regridded/'+model+'_tas_past1000_regridded.nc')
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
	cmip5_amo_box_tas.append(ts)
	coord = cube.coord('time')
	dt = coord.units.num2date(coord.points)
	years = np.array([coord.units.num2date(value).year for value in coord.points])
	cmip5_year2.append(years)


###########
#looking at subsets of the models:
###########

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
			
for i,model in enumerate(models_unique):
	pmip3_tas[model] = cmip5_amo_box_tas[i].data
	pmip3_str[model] = cmip5_max_strmfun_26[i]
	pmip3_year_str[model] = cmip5_year[i]
	pmip3_year_tas[model] = cmip5_year2[i]
	
all_models = np.unique(models+models_unique)


amo_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
amo = np.genfromtxt(amo_file, skip_header = 4)
amo_yr = amo[:,0]
amo_data = amo[:,1]
loc = np.where((amo_yr <= 1850) & (amo_yr >= 850))
amo_yr = amo_yr[loc]

amo_data = amo_data[loc]

#'CCSM4', 'CSIRO-Mk3L-1-2', 'FGOALS-gl', 'FGOALS-s2', 'GISS-E2-R',
#       'HadCM3', 'MIROC-ESM', 'MPI-ESM-P', 'MRI-CGCM3', 'bcc-csm1-1'

solar_I = ['CCSM4','FGOALS-gl', 'FGOALS-s2', 'MPI-ESM-P','bcc-csm1-1']
solar_II = ['GISS-E2-R','HadCM3','CSIRO-Mk3L-1-2']
solar_III = ['MRI-CGCM3','MIROC-ESM']


volc_I = ['bcc-csm1-1', 'FGOALS-gl', 'MRI-CGCM3','CCSM4']
volc_II = ['GISS-E2-R', 'MIROC-ESM','MPI-ESM-P','HadCM3','CSIRO-Mk3L-1-2']

plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)

for model in solar_I:
	tmp = pmip3_tas[model]
	#tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_tas[model],rm.running_mean(tmp,20),'y',alpha = 0.3,linewidth=3)
	
for model in solar_II:
	tmp = pmip3_tas[model]
	#tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_tas[model],rm.running_mean(tmp,20),'b',alpha = 0.3,linewidth=3)

for model in solar_III:
	tmp = pmip3_tas[model]
	#tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_tas[model],rm.running_mean(tmp,20),'r',alpha = 0.3,linewidth=3)

ax1.set_ylim([-0.4,0.4])

ax2 = ax1.twinx()
tmp = amo_data
#tmp = tmp/np.mean(tmp)
tmp = signal.detrend(tmp)
ax2.plot(amo_yr,tmp,'k',linewidth=3)

plt.xlim([850,1850])
plt.title('tas')
plt.show(block=False)

#################


####################

#volc composite tas

plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)


for model in volc_I:
	tmp = pmip3_tas[model]
	yr_tmp = pmip3_year_tas[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	#tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_tas[model][loc],rm.running_mean(tmp,20),'r',alpha = 0.3,linewidth=3)
	

for model in volc_II:
	tmp = pmip3_tas[model]
	yr_tmp = pmip3_year_tas[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	#tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_tas[model][loc],rm.running_mean(tmp,20),'b',alpha = 0.3,linewidth=3)


ax1.set_ylim([-0.4,0.4])

ax2 = ax1.twinx()
tmp = amo_data
#tmp = tmp/np.mean(tmp)
tmp = signal.detrend(tmp)
ax2.plot(amo_yr,tmp,'k',linewidth=3)
plt.title('volc tas')
plt.xlim([850,1850])

plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/volc_tas_amo.png')

#volc composite str

plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)


for model in volc_I:
	tmp = pmip3_str[model]
	yr_tmp = pmip3_year_str[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_str[model][loc],rm.running_mean(tmp,20),'r',alpha = 0.3,linewidth=3)
	

for model in volc_II:
	tmp = pmip3_str[model]
	yr_tmp = pmip3_year_str[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_str[model][loc],rm.running_mean(tmp,20),'b',alpha = 0.3,linewidth=3)


#ax1.set_ylim([-0.4,0.4])

ax2 = ax1.twinx()
tmp = amo_data
#tmp = tmp/np.mean(tmp)
tmp = signal.detrend(tmp)
ax2.plot(amo_yr,tmp,'k',linewidth=3)
plt.title('volc AMOC')
plt.xlim([850,1850])

plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/volc_amoc_amo.png')

####################

#solar composite tas

plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)


for model in solar_I:
	tmp = pmip3_tas[model]
	yr_tmp = pmip3_year_tas[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	#tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_tas[model][loc],rm.running_mean(tmp,20),'r',alpha = 0.3,linewidth=3)
	

for model in solar_II:
	tmp = pmip3_tas[model]
	yr_tmp = pmip3_year_tas[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	#tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_tas[model][loc],rm.running_mean(tmp,20),'b',alpha = 0.3,linewidth=3)


for model in solar_III:
	tmp = pmip3_tas[model]
	yr_tmp = pmip3_year_tas[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	#tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_tas[model][loc],rm.running_mean(tmp,20),'g',alpha = 0.3,linewidth=3)


ax1.set_ylim([-0.4,0.4])

ax2 = ax1.twinx()
tmp = amo_data
#tmp = tmp/np.mean(tmp)
tmp = signal.detrend(tmp)
ax2.plot(amo_yr,tmp,'k',linewidth=3)
plt.title('solar tas')
plt.xlim([850,1850])

plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/solar_tas_amo.png')


#solar composite str

plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)


for model in solar_I:
	tmp = pmip3_str[model]
	yr_tmp = pmip3_year_str[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_str[model][loc],rm.running_mean(tmp,20),'r',alpha = 0.3,linewidth=3)
	

for model in solar_II:
	tmp = pmip3_str[model]
	yr_tmp = pmip3_year_str[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_str[model][loc],rm.running_mean(tmp,20),'b',alpha = 0.3,linewidth=3)


for model in solar_III:
	tmp = pmip3_str[model]
	yr_tmp = pmip3_year_str[model]
	loc = np.where((yr_tmp >= 850) & (yr_tmp <= 1850))
	tmp = tmp[loc]
	tmp = tmp/np.mean(tmp)
	tmp = signal.detrend(tmp)
	ax1.plot(pmip3_year_str[model][loc],rm.running_mean(tmp,20),'g',alpha = 0.3,linewidth=3)


#ax1.set_ylim([-0.4,0.4])

ax2 = ax1.twinx()
tmp = amo_data
#tmp = tmp/np.mean(tmp)
tmp = signal.detrend(tmp)
ax2.plot(amo_yr,tmp,'k',linewidth=3)
plt.title('solar AMOC')
plt.xlim([850,1850])

plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/solar_amoc_amo.png')

####################

'''

#import pickle

#with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo.pickle', 'w') as f:
#    pickle.dump([all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,solar_I,solar_II#,solar_III], f)

with open('/home/ph290/Documents/python_scripts/pickles/palaeo_amo.pickle') as f:
    all_models,amo_yr,amo_data,pmip3_str,pmip3_year_str,pmip3_tas,pmip3_year_tas,solar_I,solar_II,solar_III = pickle.load(f)

all_models = ['bcc-csm1-1', 'MRI-CGCM3','CCSM4','GISS-E2-R', 'MIROC-ESM','MPI-ESM-P','HadCM3','CSIRO-Mk3L-1-2']

solar_I = ['CCSM4','FGOALS-gl', 'FGOALS-s2', 'MPI-ESM-P','bcc-csm1-1']
solar_II = ['GISS-E2-R','HadCM3','CSIRO-Mk3L-1-2']
solar_III = ['MRI-CGCM3','MIROC-ESM']
solar_not = ['CCSM4','MPI-ESM-P','bcc-csm1-1','MRI-CGCM3','MIROC-ESM']

volc_I = ['bcc-csm1-1', 'FGOALS-gl', 'MRI-CGCM3','CCSM4']
volc_II = ['GISS-E2-R', 'MIROC-ESM','MPI-ESM-P','HadCM3','CSIRO-Mk3L-1-2']


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


################
#calculate means.........
################

all_years = np.linspace(850,1850,(1851-850))
average_tas_solar_III = np.empty([np.size(all_years),np.size(solar_III)])
average_tas_solar_III[:] = np.NAN
average_tas_solar_II = np.empty([np.size(all_years),np.size(solar_II)])
average_tas_solar_II[:] = np.NAN
average_tas_solar_I = np.empty([np.size(all_years),np.size(solar_I)])
average_tas_solar_I[:] = np.NAN

average_str_solar_III = np.empty([np.size(all_years),np.size(solar_III)])
average_str_solar_III[:] = np.NAN
average_str_solar_II = np.empty([np.size(all_years),np.size(solar_II)])
average_str_solar_II[:] = np.NAN
average_str_solar_I = np.empty([np.size(all_years),np.size(solar_I)])
average_str_solar_I[:] = np.NAN

average_tas_volc_II = np.empty([np.size(all_years),np.size(volc_II)])
average_tas_volc_II[:] = np.NAN
average_tas_volc_I = np.empty([np.size(all_years),np.size(volc_I)])
average_tas_volc_I[:] = np.NAN

average_str_volc_II = np.empty([np.size(all_years),np.size(volc_II)])
average_str_volc_II[:] = np.NAN
average_str_volc_I = np.empty([np.size(all_years),np.size(volc_I)])
average_str_volc_I[:] = np.NAN

smoothing_val = 10

counter = -1
for i,model in enumerate(solar_III):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_solar_III[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_solar_III[j,counter] = data[loc2]
		except:
			None


counter = -1
for i,model in enumerate(solar_II):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_solar_II[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_solar_II[j,counter] = data[loc2]
		except:
			None


counter = -1
for i,model in enumerate(solar_I):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_solar_I[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_solar_I[j,counter] = data[loc2]
		except:
			None

counter = -1
for i,model in enumerate(volc_I):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_volc_I[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_volc_I[j,counter] = data[loc2]
		except:
			None

counter = -1
for i,model in enumerate(volc_II):
	counter += 1
	tmp = pmip3_tas[model]
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= 1850) & (pmip3_year_tas[model] >= 850))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_tas_volc_II[j,counter] = data[loc2]
		except:
			None
	tmp = pmip3_str[model]
	tmp = tmp/np.mean(tmp)
	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= 1850) & (pmip3_year_str[model] >= 850))
	tmp = tmp[loc]
	yrs =pmip3_year_str[model][loc]
	data = signal.detrend(tmp)
	for j,yr in enumerate(all_years):
		try:
			loc2 = np.where(yrs == yr)
			average_str_volc_II[j,counter] = data[loc2]
		except:
			None


average_tas_solar_IIIb = np.mean(average_tas_solar_III,axis = 1)
average_tas_solar_IIb = np.mean(average_tas_solar_II,axis = 1)
average_tas_solar_Ib = np.mean(average_tas_solar_I,axis = 1)

average_str_solar_IIIb = np.mean(average_str_solar_III,axis = 1)
average_str_solar_IIb = np.mean(average_str_solar_II,axis = 1)
average_str_solar_Ib = np.mean(average_str_solar_I,axis = 1)

average_tas_volc_IIb = np.mean(average_tas_volc_II,axis = 1)
average_tas_volc_Ib = np.mean(average_tas_volc_I,axis = 1)

average_str_volc_IIb = np.mean(average_str_volc_II,axis = 1)
average_str_volc_Ib = np.mean(average_str_volc_I,axis = 1)

#plotting means



N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
#low_cutoff=150.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
#high_cutoff=150.0 #years: 1 for filtering at 1 yr. 5 for filtering at 5 years
#middle_cuttoff_low=1.0
################
middle_cuttoff_high=100.0
################

#Wn_low=timestep_between_values/low_cutoff
#Wn_high=timestep_between_values/high_cutoff
#Wn_mid_low=timestep_between_values/middle_cuttoff_low
Wn_mid_high=timestep_between_values/middle_cuttoff_high

#b, a = scipy.signal.butter(N, Wn_low, btype='low')
#b1, a1 = scipy.signal.butter(N, Wn_mid_low, btype='low')
b2, a2 = scipy.signal.butter(N, Wn_mid_high, btype='high')
#data = scipy.signal.filtfilt(b2, a2, tmp)



alph = 0.2
wdth = 5

#solar tas
plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)

# ax1.plot(all_years,average_tas_solar_IIIb,'r',alpha = alph,linewidth=wdth)
# ax1.plot(all_years,average_tas_solar_IIb,'b',alpha = alph,linewidth=wdth)
# ax1.plot(all_years,average_tas_solar_Ib,'g',alpha = alph,linewidth=wdth)

loc = np.where(np.logical_not(np.isnan(average_tas_solar_IIIb)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_tas_solar_IIIb[loc]),'r',alpha = alph,linewidth=wdth)
loc = np.where(np.logical_not(np.isnan(average_tas_solar_IIb)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_tas_solar_IIb[loc]),'b',alpha = alph,linewidth=wdth)
loc = np.where(np.logical_not(np.isnan(average_tas_solar_Ib)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_tas_solar_Ib[loc]),'g',alpha = alph,linewidth=wdth)

ax1.set_ylim([-0.5,0.3])

ax2 = ax1.twinx()
tmp = amo_data
tmp = signal.detrend(tmp)
#ax2.plot(amo_yr,tmp,'k',linewidth=wdth)
ax2.plot(amo_yr,scipy.signal.filtfilt(b2, a2, tmp),'k',linewidth=wdth,alpha = 0.9)
ax3 = ax2.twinx()
ax3.plot(voln_n[:,0],voln_n[:,1],'k',alpha = 0.3)
plt.title('solar tas')
plt.xlim([850,1850])

#plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/solar_tas_amob.png')

#volc tas
plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)

#ax1.plot(all_years,average_tas_volc_IIb,'b',alpha = alph,linewidth=wdth)
#ax1.plot(all_years,average_tas_volc_Ib,'g',alpha = alph,linewidth=wdth)

loc = np.where(np.logical_not(np.isnan(average_tas_volc_IIb)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_tas_volc_IIb[loc]),'b',alpha = alph,linewidth=wdth)
loc = np.where(np.logical_not(np.isnan(average_tas_volc_Ib)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_tas_volc_Ib[loc]),'g',alpha = alph,linewidth=wdth)

ax1.set_ylim([-0.5,0.3])

ax2 = ax1.twinx()
tmp = amo_data
tmp = signal.detrend(tmp)
#ax2.plot(amo_yr,tmp,'k',linewidth=3)
ax2.plot(amo_yr,scipy.signal.filtfilt(b2, a2, tmp),'k',linewidth=wdth,alpha = 0.9)
ax3 = ax2.twinx()
ax3.plot(voln_n[:,0],voln_n[:,1],'k',alpha = 0.3)
plt.title('volc tas')
plt.xlim([850,1850])

#plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/volc_tas_amob.png')


#solar str
plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)

#ax1.plot(all_years,average_str_solar_IIIb,'r',alpha = alph,linewidth=wdth)
#ax1.plot(all_years,average_str_solar_IIb,'b',alpha = alph,linewidth=wdth)
#ax1.plot(all_years,average_str_solar_Ib,'g',alpha = alph,linewidth=wdth)

loc = np.where(np.logical_not(np.isnan(average_str_solar_IIIb)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_IIIb[loc]),'r',alpha = alph,linewidth=wdth/2.0)
loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc]),'b',alpha = alph*2.0,linewidth=wdth)
loc = np.where(np.logical_not(np.isnan(average_str_solar_Ib)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_Ib[loc]),'g',alpha = alph,linewidth=wdth/2.0)

ax1.set_ylim([-0.06,0.06])

ax2 = ax1.twinx()
tmp = amo_data
tmp = signal.detrend(tmp)
#ax2.plot(amo_yr,tmp,'k',linewidth=3)
ax2.plot(amo_yr,scipy.signal.filtfilt(b2, a2, tmp),'k',linewidth=wdth,alpha = 0.9)
ax3 = ax2.twinx()
ax3.plot(voln_n[:,0],voln_n[:,1],'k',alpha = 0.3)
plt.title('solar str')
plt.xlim([850,1850])

#plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/solar_str_amob.png')

#volc str
plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)

#ax1.plot(all_years,average_str_volc_IIb,'b',alpha = alph,linewidth=wdth)
#ax1.plot(all_years,average_str_volc_Ib,'g',alpha = alph,linewidth=wdth)

loc = np.where(np.logical_not(np.isnan(average_str_volc_IIb)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_volc_IIb[loc]),'b',alpha = alph,linewidth=wdth)
loc = np.where(np.logical_not(np.isnan(average_str_volc_Ib)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_volc_Ib[loc]),'g',alpha = alph,linewidth=wdth)

ax1.set_ylim([-0.06,0.06])

ax2 = ax1.twinx()
tmp = amo_data
tmp = signal.detrend(tmp)
#ax2.plot(amo_yr,tmp,'k',linewidth=3)
ax2.plot(amo_yr,scipy.signal.filtfilt(b2, a2, tmp),'k',linewidth=wdth,alpha = 0.9)
ax3 = ax2.twinx()
ax3.plot(voln_n[:,0],voln_n[:,1],'k',alpha = 0.3)
plt.title('volc str')
plt.xlim([850,1850])

#plt.show(block=False)
plt.savefig('/home/ph290/Documents/figures/volc_str_amob.png')


#paper plots:

'''
#read in volc data
'''



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

end_year = 1849
start_year = 851

amo_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
amo = np.genfromtxt(amo_file, skip_header = 4)
amo_yr = amo[:,0]
amo_data = amo[:,1]
loc = np.where((amo_yr <= 1850) & (amo_yr >= start_year))
amo_yr = amo_yr[loc]
amo_data = amo_data[loc]
amo_data = scipy.signal.filtfilt(b2, a2, amo_data)
x = amo_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
amo_data = x



smoothing_val=10
wdth = 2

plt.close('all')
fig = plt.figure(figsize=(8,12),dpi=80)
ax1 = fig.add_subplot(311)

mean_data = np.zeros([1+end_year-start_year,np.size(all_models)])

for i,model in enumerate(all_models):
	tmp = pmip3_tas[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_tas[model] <= end_year) & (pmip3_year_tas[model] >= start_year))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	l1 = ax1.plot(yrs,rm.running_mean(data3,smoothing_val),'b',alpha = 0.1,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble member')
	mean_data[:,i] = data3
	
mean_data2 = np.mean(mean_data, axis = 1)
l2 = ax1.plot(yrs,mean_data2,'b',linewidth=wdth,alpha=0.9,label = 'CMIP5/PMIP3 ensemble mean')
ax1.set_ylabel('Atlantic temperature anomaly ($^o$C)')

ax2 = ax1.twinx()
l3 = ax2.plot(amo_yr,amo_data,'k',linewidth=wdth,alpha=0.7,label = 'AMV index (Mann et al., 2009)')
ax2.set_ylim([-0.4,1.2])
ax2.set_ylabel('AMV index')

lns = l1+l2+l3
 
#fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.5,prop={'size':8})

# ax3 = ax2.twinx()
# ax3.plot(voln_n[:,0],voln_n[:,1],'k',alpha=0.3)
# ax3.set_axis_off()

###
#linear model
###


ax4 = fig.add_subplot(312)

smth = 7
smth2 = 7

#end
start_year = 860
end_year = 1840
volc_year = voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,0]
vns = running_mean_post.running_mean_post(voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,1],smth*36.0)
#loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
#tmp = running_mean_post.running_mean_post(mean_data_str_subset2,smoothing_val)
#loc = np.where(np.logical_not(np.isnan(tmp)))
# best_str = rm.running_mean(scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc]),smth2)
#best_str = tmp[loc]
#srm_func = np.interp(volc_year,yrs[loc],best_str)

'''
#y2 = amo_data.copy()
#mann_amo = np.interp(volc_year,amo_yr,y2)
#y = mann_amo
y2 = mean_data2.copy()
model_amo = np.interp(volc_year,yrs,y2)
y = model_amo
#stack explanatory variables into an array
x1 = vns
x2 = srm_func
x = np.column_stack((x1,x2))
x = sm.add_constant(x)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

l1 = ax4.plot(volc_year,y,'b',linewidth=wdth,alpha=0.7,label = 'CMIP5/PMIP3 AMV index')
l2 = ax4.plot(volc_year,results.params[2]*x2+results.params[1]*x1+results.params[0],'r',linewidth=wdth,alpha=0.7, label = 'statistical model of AMV based on volcanic index')
'''

#y2 = amo_data.copy()
#mann_amo = np.interp(volc_year,amo_yr,y2)
#y = mann_amo
y2 = mean_data2.copy()
model_amo = np.interp(volc_year,yrs,y2)
y = model_amo
#stack explanatory variables into an array
x1 = vns
x = sm.add_constant(x1)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

l1 = ax4.plot(volc_year,y,'b',linewidth=wdth,alpha=0.7,label = 'CMIP5/PMIP3 AMV index')
l2 = ax4.plot(volc_year,results.params[1]*x1+results.params[0],'r',marker ='o', markersize=2,markevery=10*36,linewidth=wdth,alpha=0.7, label = 'statistical model of CMIP5/PMIP3 AMV based on volcanic index')

ax4.set_ylim([0.2,0.9])

'''
#middle2
start_year = 1300
end_year = 1400
volc_year = voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,0]
vns = running_mean_post.running_mean_post(voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,1],smth*36.0)
#loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
tmp = running_mean_post.running_mean_post(mean_data_str2,smoothing_val)
loc = np.where(np.logical_not(np.isnan(tmp)))
# best_str = rm.running_mean(scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc]),smth2)
best_str = tmp[loc]
srm_func = np.interp(volc_year,yrs[loc],best_str)

y2 = amo_data.copy()
mann_amo = np.interp(volc_year,amo_yr,y2)
y = mann_amo
#stack explanatory variables into an array
x1 = vns
x2 = srm_func
x = np.column_stack((x1,x2))
x = sm.add_constant(x)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

l1 = ax4.plot(volc_year,y,'k',linewidth=wdth,alpha=0.7,label = 'AMV index (Mann et al., 2009)')
# l2 = ax4.plot(volc_year,results.params[2]*x2+results.params[1]*x1+results.params[0],'g',linewidth=wdth,alpha=0.7, label = 'statistical model of AMV based on volcanic index')

y2 = amo_data.copy()
mann_amo = np.interp(volc_year,amo_yr,y2)
y = mann_amo
#stack explanatory variables into an array
x1 = vns
x = sm.add_constant(x1)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

# l2 = ax4.plot(volc_year,results.params[1]*x1+results.params[0],'g:',linewidth=wdth,alpha=0.7, label = 'statistical model of AMV based on volcanic index')


#middle1
start_year = 1200
end_year = 1300
volc_year = voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,0]
vns = running_mean_post.running_mean_post(voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,1],smth*36.0)
#loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
tmp = running_mean_post.running_mean_post(mean_data_str2,smoothing_val)
loc = np.where(np.logical_not(np.isnan(tmp)))
# best_str = rm.running_mean(scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc]),smth2)
best_str = tmp[loc]
srm_func = np.interp(volc_year,yrs[loc],best_str)

y2 = amo_data.copy()
mann_amo = np.interp(volc_year,amo_yr,y2)
y = mann_amo
#stack explanatory variables into an array
x1 = vns
x2 = srm_func
x = np.column_stack((x1,x2))
x = sm.add_constant(x)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

l1 = ax4.plot(volc_year,y,'k',linewidth=wdth,alpha=0.7,label = 'AMV index (Mann et al., 2009)')
# l2 = ax4.plot(volc_year,results.params[2]*x2+results.params[1]*x1+results.params[0],'b',linewidth=wdth,alpha=0.7, label = 'statistical model of AMV based on volcanic index')

y2 = amo_data.copy()
mann_amo = np.interp(volc_year,amo_yr,y2)
y = mann_amo
#stack explanatory variables into an array
x1 = vns
x = sm.add_constant(x1)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

# l2 = ax4.plot(volc_year,results.params[1]*x1+results.params[0],'b:',linewidth=wdth,alpha=0.7, label = 'statistical model of AMV based on volcanic index')


#start
start_year = 870
# start_year = 950
end_year = 1200
volc_year = voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,0]
vns = running_mean_post.running_mean_post(voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,1],smth*36.0)
#loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
tmp = running_mean_post.running_mean_post(mean_data_str2,smoothing_val)
loc = np.where(np.logical_not(np.isnan(tmp)))
# best_str = rm.running_mean(scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc]),smth2)
best_str = tmp[loc]
srm_func = np.interp(volc_year,yrs[loc],best_str)

y2 = amo_data.copy()
mann_amo = np.interp(volc_year,amo_yr,y2)
y = mann_amo
#stack explanatory variables into an array
x1 = vns
x2 = srm_func
x = np.column_stack((x1,x2))
x = sm.add_constant(x)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

l1 = ax4.plot(volc_year,y,'k',linewidth=wdth,alpha=0.7,label = 'AMV index (Mann et al., 2009)')
l2 = ax4.plot(volc_year,results.params[2]*x2+results.params[1]*x1+results.params[0],color = 'orange',linewidth=wdth,alpha=0.7, label = 'statistical model of AMV based on volcanic index and CMIP5 AMOC strength')


y2 = amo_data.copy()
mann_amo = np.interp(volc_year,amo_yr,y2)
y = mann_amo
#stack explanatory variables into an array
x1 = vns
x = sm.add_constant(x1)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

from matplotlib.lines import Line2D
l2b = ax4.plot(volc_year,results.params[1]*x1+results.params[0],color = 'orange',marker ='o', markersize=2,markevery=10*36,alpha=0.7, label = 'statistical model of AMV based on volcanic index')

'''

ax5 = ax4.twinx()
l3 = ax5.plot(voln_n[:,0],voln_n[:,1],'k',alpha=0.5,label = 'Volcanic index (Crowley and Unterman. 2012)')
ax5.set_label('Aerosol Optical Depth')
# ax5.set_axis_off()

lns = l1+l2
#fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'upper right', fancybox=True, framealpha=0.5,prop={'size':8})



###
#AMOC
###

mean_data_str = np.zeros([1+end_year-start_year,np.size(all_models)])
mean_data_str[:] = np.NAN
mean_data_str_subset = np.zeros([1+end_year-start_year,np.size(solar_II)])
mean_data_str_subset[:] = np.NAN

mean_data_str_subsetb = np.zeros([1+end_year-start_year,np.size(solar_I)])
mean_data_str_subsetb[:] = np.NAN

mean_data_str_subsetc = np.zeros([1+end_year-start_year,np.size(solar_III)])
mean_data_str_subsetc[:] = np.NAN

for i,model in enumerate(all_models):
	print model
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	for j,yr_tmp in enumerate(range(start_year,end_year)):
		loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
		if np.size(loc2) > 0:
			mean_data_str[j,i] = data3[loc2]
	
mean_data_str2 = np.mean(mean_data_str, axis = 1)

for i,model in enumerate(solar_II):
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs_subset = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	for j,yr_tmp in enumerate(range(start_year,end_year)):
		loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
		if np.size(loc2) > 0:
			mean_data_str_subset[j,i] = data3[loc2]

mean_data_str_subset2 = np.mean(mean_data_str_subset, axis = 1)

for i,model in enumerate(solar_I):
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs_subset = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	for j,yr_tmp in enumerate(range(start_year,end_year)):
		loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
		if np.size(loc2) > 0:
			mean_data_str_subsetb[j,i] = data3[loc2]

mean_data_str_subset2b = np.mean(mean_data_str_subsetb, axis = 1)


for i,model in enumerate(solar_III):
	tmp = pmip3_str[model]
# 	tmp = rm.running_mean(tmp,smoothing_val)
	loc = np.where((np.logical_not(np.isnan(tmp))) & (pmip3_year_str[model] <= end_year) & (pmip3_year_str[model] >= start_year))
	tmp = tmp[loc]
	yrs_subset = pmip3_year_tas[model][loc]
	data2 = scipy.signal.filtfilt(b2, a2, tmp)
	x = data2
	data3 = (x-np.min(x))/(np.max(x)-np.min(x))
	for j,yr_tmp in enumerate(range(start_year,end_year)):
		loc2 = np.where(pmip3_year_str[model][loc] == yr_tmp)
		if np.size(loc2) > 0:
			mean_data_str_subsetc[j,i] = data3[loc2]

mean_data_str_subset2c = np.mean(mean_data_str_subsetc, axis = 1)





ax20 = fig.add_subplot(313)

#solar str

# loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
# best_str = scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc])
# 
# loc = np.where(np.logical_not(np.isnan(average_str_solar_IIIb)))
# ax20.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_IIIb[loc]),'r',alpha = alph,linewidth=wdth/2.0)
# loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
# ax20.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc]),'b',alpha = alph*2.0,linewidth=wdth)
# loc = np.where(np.logical_not(np.isnan(average_str_solar_Ib)))
# ax20.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_Ib[loc]),'g',alpha = alph,linewidth=wdth/2.0)

smoothing_val = 5

for i,dummy in enumerate(all_models):
	l1 = ax20.plot(yrs,rm.running_mean(mean_data_str[:,i],smoothing_val),'g',alpha = 0.1,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble member AMOC')

l2 = ax20.plot(yrs,rm.running_mean(mean_data_str2,smoothing_val),'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean AMOC')

#l2 = ax20.plot(yrs,running_mean_post.running_mean_post(mean_data_str_subset2b,smoothing_val),'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean')
#l2 = ax20.plot(yrs,running_mean_post.running_mean_post(mean_data_str_subset2c,smoothing_val),'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean')

#for i,dummy in enumerate(solar_II):
#	l1 = ax20.plot(yrs,running_mean_post.running_mean_post(mean_data_str_subset[:,i],smoothing_val),'g',alpha = 0.1,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble member')

#l2 = ax20.plot(yrs,running_mean_post.running_mean_post(mean_data_str_subset2,smoothing_val),'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean')
ax20.set_ylim([0.3,0.5])
ax20.set_ylabel('AMOC strength')
# ax20.plot(yrs_subset,rm.running_mean(mean_data_str_subset2,smoothing_val),'g',alpha = 0.7,linewidth=wdth)


# ax20.set_ylim([-0.06,0.06])

# ax2  = ax1.twinx()
# tmp = amo_data
# tmp = signal.detrend(tmp)
# #ax2.plot(amo_yr,tmp,'k',linewidth=3)
# ax2.plot(amo_yr,scipy.signal.filtfilt(b2, a2, tmp),'k',linewidth=wdth,alpha = 0.9)

ax21 = ax20.twinx()
l3 = ax21.plot(amo_yr,amo_data,'k',linewidth=wdth,alpha=0.7,label = 'AMV index (Mann et al., 2009)')
ax21.set_ylim([-0.2,1.2])
ax21.set_ylabel('AMV index')

lns = l1+l2+l3
 
#fig.legend((l3, l1, l2),('AMV index (Mann et al., 2009)','CMIP5/PMIP3 ensemble member','CMIP5/PMIP3 ensemble mean'))
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.5,prop={'size':8})


plt.xlim([850,1850])
ax1.set

'''

###
#AMOC_obs
###

N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
middle_cuttoff_high=100.0

Wn_mid_high=timestep_between_values/middle_cuttoff_high

b, a = scipy.signal.butter(N, Wn_mid_high, btype='high')


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

smoothing = 5
alph_val = 0.75


ax4 = fig.add_subplot(313)
ax4.plot(amo_yr,amo_data,'k',linewidth = 3,alpha=alph_val)
ax5 = ax4.twinx()
ax5.plot(e_europe_yr,rm.running_mean(colarado_canada_data,smoothing_val)-rm.running_mean(iceland_e_europe_data,smoothing_val),'r',linewidth = 3,alpha=alph_val)
ax31 = ax5.twinx()
ax31.plot(yrs,rm.running_mean(mean_data_str2,smoothing_val),'g',alpha = 0.9,linewidth=wdth,label = 'CMIP5/PMIP3 ensemble mean')

'''


###
#linear_model2
###

start_date = 850
end_date = 1850

N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
low_cutoff=100.0
high_cutoff=7.0

Wn_low=timestep_between_values/low_cutoff
Wn_high=timestep_between_values/high_cutoff

b, a = scipy.signal.butter(N, Wn_low, btype='high')
b1, a1 = scipy.signal.butter(N, Wn_high, btype='low')

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


directory = '/home/ph290/data0/misc_data/last_millenium_solar/'
    
file1 = directory+'tsi_VK.txt'
file3 = directory+'tsi_SBF_11yr.txt' 
file4 = directory+'tsi_DB_lin_40_11yr.txt' 


data1 = np.genfromtxt(file1,skip_header = 4)
data1_yr = data1[:,0]
loc = np.where((data1_yr <= end_date) & (data1_yr >= start_date))
data1_yr = data1[loc[0],0]
data1_data = data1[loc[0],1]
data1_data = scipy.signal.filtfilt(b, a, data1_data)
x = data1_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
data1_data = x


data3 = np.genfromtxt(file3,skip_header = 4)
data3_yr = data3[:,0]
loc = np.where((data3_yr <= end_date) & (data3_yr >= start_date))
data3_yr = data3[loc[0],0]
data3_data = data3[loc[0],1]
data3_data = scipy.signal.filtfilt(b, a, data3_data)
x = data3_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
data3_data = x

data4 = np.genfromtxt(file4,skip_header = 4)
data4_yr = data4[:,0]
loc = np.where((data4_yr <= end_date) & (data4_yr >= start_date))
data4_yr = data4[loc[0],0]
data4_data = data4[loc[0],1]
data4_data = scipy.signal.filtfilt(b, a, data4_data)
x = data4_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
data4_data = x

smth =7
smth2 =1
volc_yr = voln_n[:,0]
loc_v = np.where((volc_yr <= end_date) & (volc_yr >= start_date))
volc_yr_II = volc_yr[loc_v]

tmp = data1_data
tmp_yr = data1_yr
tmp = scipy.signal.filtfilt(b, a, tmp)
tmp = scipy.signal.filtfilt(b1, a1, tmp)

data1_int = np.interp(volc_yr_II,tmp_yr,tmp)
#data1_int = rmp.running_mean_post(data1_int,smth2*36.0)

tmp = data3_data
tmp_yr = data3_yr
tmp = scipy.signal.filtfilt(b, a, tmp)
tmp = scipy.signal.filtfilt(b1, a1, tmp)

data3_int = np.interp(volc_yr_II,tmp_yr,tmp)
#data3_int = rmp.running_mean_post(data3_int,smth2*36.0)

tmp = data4_data
tmp_yr = data4_yr
tmp = scipy.signal.filtfilt(b, a, tmp)
tmp = scipy.signal.filtfilt(b1, a1, tmp)

data4_int = np.interp(volc_yr_II,tmp_yr,tmp)
#data4_int = rmp.running_mean_post(data4_int,smth2*36.0)

#
#data1_int[np.where(np.isnan(data1_int))] = 0.0
smth = 7
smth2 = 1
#smoothing_val
tmp2 = rm.running_mean(mean_data_str2,5)
loc = np.where((np.logical_not(np.isnan(tmp2))) & (yrs >= start_date) & (yrs <= end_date))
tmp2 = tmp2[loc]
tmp_yr = yrs[loc]
model_str = np.interp(volc_yr_II,tmp_yr,tmp2)
vns = rmp.running_mean_post(voln_n[loc_v[0],1],smth*36.0)

mean_solar = np.zeros([np.size(data1_int),3])
mean_solar[:,0] = data1_int
mean_solar[:,1] = data3_int
mean_solar[:,2] = data4_int
mean_solar = np.mean(mean_solar,axis = 1)

x1 = vns
#x2 = rmp.running_mean_post(data3_int,smth2*36.0)
x2 = rmp.running_mean_post(mean_solar,smth2*36)
#rmp.running_mean_post(data1_int,smth2*36)
#x3 = rmp.running_mean_post(data3_int,smth2*36)
#x4 = rmp.running_mean_post(data4_int,smth2*36)
y = model_str
x = np.column_stack((x1,x2))
#stack explanatory variables into an array
x = sm.add_constant(x)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

#multi model temperature
#yrs,mean_data2

'''
ax41 = fig.add_subplot(414)

r_data_file = '/home/ph290/data0/reynolds/ultra_data.csv'
r_data = np.genfromtxt(r_data_file,skip_header = 1,delimiter = ',')

tmp = r_data[:,1]

smoothing_val = 5

tmp = rm.running_mean(tmp,smoothing_val)
loc = np.where((np.logical_not(np.isnan(tmp))) & (r_data[:,0] >= start_date) & (r_data[:,0] <= end_date))
tmp = tmp[loc]
tmp_yr = r_data[loc[0],0]
tmp = scipy.signal.filtfilt(b, a, tmp)

l1 = ax41.plot(tmp_yr,tmp,'r',linewidth = 2,alpha = 0.75,label = 'Reynolds d18O')

#ax41.plot(volc_yr_II,results.params[2]*x2+results.params[1]*x1+results.params[0],'r',linewidth = 3,alpha = 0.75)


ax42 = ax41.twinx()
tmp2 = rm.running_mean(mean_data_str2,7)
loc = np.where((np.logical_not(np.isnan(tmp2))) & (yrs >= start_date) & (yrs <= end_date))
tmp2 = tmp2[loc]
tmp_yr = yrs[loc]
#ax42.plot(tmp_yr,tmp2,'g',linewidth = 2,alpha = 0.75)

l2 = ax42.plot(yrs,rm.running_mean(mean_data_str2,smoothing_val),'g',linewidth = 2,alpha = 0.75,label = 'CMIP5/PMIP3 ensemble mean AMOC')
ax42.set_ylim([0.3,0.5])

lns = l1+l2
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.5,prop={'size':8})

ax41.set_ylabel('N. Iceland d18O')
ax42.set_ylabel('AMOC Strength')
ax42.set_xlabel('Year')


#ax42.plot(volc_yr_II,data1_int,'y',linewidth = 3,alpha = 0.5)
#ax42.plot(volc_yr_II,x2,'r',linewidth = 3,alpha = 0.5)
#ax42.plot(volc_yr_II,data4_int,'b',linewidth = 3,alpha = 0.5)

'''

###
#end
###

ax1.set_xlim([950,1850])
ax2.set_xlim([950,1850])
ax3.set_xlim([950,1850])
ax4.set_xlim([950,1850])
ax5.set_xlim([950,1850])
ax20.set_xlim([950,1850])
#ax41.set_xlim([950,1850])
ax4.set_ylabel('AMV index')


plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/palaeoamo/pmip3_tas_and_stat_modelled_amo.png')
	
	
###############
###############

'''

#solar str
plt.close('all')
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)

#ax1.plot(all_years,average_str_solar_IIIb,'r',alpha = alph,linewidth=wdth)
#ax1.plot(all_years,average_str_solar_IIb,'b',alpha = alph,linewidth=wdth)
#ax1.plot(all_years,average_str_solar_Ib,'g',alpha = alph,linewidth=wdth)

loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
best_str = scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc])

loc = np.where(np.logical_not(np.isnan(average_str_solar_IIIb)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_IIIb[loc]),'r',alpha = alph,linewidth=wdth/2.0)
loc = np.where(np.logical_not(np.isnan(average_str_solar_IIb)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_IIb[loc]),'b',alpha = alph*2.0,linewidth=wdth)
loc = np.where(np.logical_not(np.isnan(average_str_solar_Ib)))
ax1.plot(all_years[loc],scipy.signal.filtfilt(b2, a2, average_str_solar_Ib[loc]),'g',alpha = alph,linewidth=wdth/2.0)

ax1.set_ylim([-0.06,0.06])

ax2 = ax1.twinx()
tmp = amo_data
tmp = signal.detrend(tmp)
#ax2.plot(amo_yr,tmp,'k',linewidth=3)
ax2.plot(amo_yr,scipy.signal.filtfilt(b2, a2, tmp),'k',linewidth=wdth,alpha = 0.9)

plt.xlim([850,1850])
ax1.set

plt.show(block=False)
#plt.savefig('/home/ph290/Documents/figures/solar_str_amob.png')
	
	
	
'''
	
	
	
	
	
	
	
	
	
	
	
	
	
