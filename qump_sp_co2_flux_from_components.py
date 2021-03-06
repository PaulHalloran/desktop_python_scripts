import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
import iris.analysis.cartography
import iris.coord_categorisation
import iris.analysis
import time
import matplotlib as mpl
import running_mean as run_mean
import running_mean_post as rm2
import iris.plot as iplt
import matplotlib.cm as mpl_cm
import carbchem_cube
import scipy
import scipy.signal
import pickle
from scipy.stats import gaussian_kde



'''
#We will use the following functions, so make sure they are available
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




def extract_region(cube):
    lon_west = -60.0
    lon_east = -10
    lat_south = 48
    lat_north = 66.0 
    try:
        cube.coord('longitude').guess_bounds()
        cube.coord('latitude').guess_bounds()
    except:
        None
    cube = cube.intersection(longitude=(lon_west, lon_east))
    cube = cube.intersection(latitude=(lat_south, lat_north))
    return cube


def extract_region_stg(cube):
    lon_west = -60.0
    lon_east = -10
    lat_south = -30
    lat_north = 30
    try:
        cube.coord('longitude').guess_bounds()
        cube.coord('latitude').guess_bounds()
    except:
        None
    cube = cube.intersection(longitude=(lon_west, lon_east))
    cube = cube.intersection(latitude=(lat_south, lat_north))
    return cube


def load_and_agregate_cube(run,a,b,c):
    a = str(a)
    b = str(b)
    c = str(c)
    cube = iris.load_cube(run+'/*'+a+'.'+b+'.'+c+'*.pp',iris.AttributeConstraint(STASH='m'+a+'s'+b+'i'+c),callback=my_callback)
    iris.coord_categorisation.add_year(cube, 'time', name='year2')
    return cube.aggregated_by('year2', iris.analysis.MEAN)


def area_average(cube):
    grid_areas = iris.analysis.cartography.area_weights(cube)
    return cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)


f = open('/home/ph290/data0/misc_data/qump_run_names.txt','r')
name_data = f.read()
f.close()

lines = name_data.split('\n')[1:-1]

all_names = np.chararray([3,np.size(lines)],5)

for i,line in enumerate(lines):
    tmp = line.split(',')
    all_names[0,i] = tmp[2]
    all_names[1,i] = tmp[5]
    all_names[2,i] = tmp[7]


#time.sleep(60.0*60.0*12*2)

def my_callback(cube, field, filename):
        cube.remove_coord('forecast_reference_time')
        cube.remove_coord('forecast_period')
        #the cubes were not merging properly before, because the time coordinate appeard to have teo different names... I think this may work

directory = '/data/data1/ph290/qump_co2/stash_split/qump_n_atl_mor_var_monthly_ss/'
output_directory = ('/home/ph290/data1/qump_co2/global_avg/')

runs = glob.glob(directory+'/?????')

run_names = []
stg_fgco2 = []
spg_fgco2 = []
spg_pco2 = []
spg_pco2_sst_const = []
spg_pco2_sss_const = []
spg_pco2_tco2_const = []
spg_pco2_talk_const = []
run_year = []
atm_co2 = []
spg_tco2 = []
spg_talk = []
spg_sst = []
spg_sss = []

run = runs[0]
sst = extract_region(load_and_agregate_cube(run,'02','00','101')+273.15)
mdi = sst.data.fill_value

sizing = [0,91,91+61,91+61+96]

'''

for i,dummy in enumerate(all_names[0,:]):
    print i
    temp_array0 = np.zeros(91+61+96)
    temp_array1 = np.zeros(91+61+96)
    temp_array2 = np.zeros(91+61+96)
    temp_array3 = np.zeros(91+61+96)
    temp_array4 = np.zeros(91+61+96)
    temp_array5 = np.zeros(91+61+96)
    temp_array6 = np.zeros(91+61+96)
    temp_array7 = np.zeros(91+61+96)
    for j,run_tmp in enumerate(all_names[:,i]):
        print j
        #if i >= 114:
        #run_name = run.split('/')[7]
        #run_names.append(run_name)
        tmp = glob.glob(directory+'/'+run_tmp)
        if np.size(tmp) > 0:
            run = tmp[0]
            cube = load_and_agregate_cube(run,'02','30','249')
            time_mean1 = area_average(extract_region(cube))
            time_mean2 = area_average(extract_region_stg(cube))
            sst = extract_region(load_and_agregate_cube(run,'02','00','101')+273.15)
            sss = extract_region(load_and_agregate_cube(run,'02','00','102')*1000+35.0)
            tco2 = extract_region(load_and_agregate_cube(run,'02','00','103')/(1026.*1000.)*1.0e3)
            talk = extract_region(load_and_agregate_cube(run,'02','00','104')/(1026.*1000.)*1.0e3)
            co2 = carbchem_cube.carbchem(1,mdi,sst,sss,tco2,talk)
            #spg_pco2.append(area_average(co2).data)
            #homogenise_sst
            if j == 0:
                #just using the initial value from the start of the historical run
                sst_tmp = sst.copy()
            sst2 = sst.copy()
            sst2.data = np.tile(sst_tmp[0].data,[np.shape(sst2)[0],1,1])
            co2_sst_const = carbchem_cube.carbchem(1,mdi,sst2,sss,tco2,talk)
            #spg_pco2_sst_const.append(area_average(co2_sst_const).data)
            #homogenise_sss
            if j == 0:
                sss_tmp = sss.copy()
            sss2 = sss.copy()
            sss2.data = np.tile(sss_tmp[0].data,[np.shape(sss2)[0],1,1])
            co2_sss_const = carbchem_cube.carbchem(1,mdi,sst,sss2,tco2,talk)
            #spg_pco2_sss_const.append(area_average(co2_sss_const).data)
            #homogenise_tco2
            if j == 0:
                tco2_tmp = tco2.copy()
            tco22 = tco2.copy()
            tco22.data = np.tile(tco2_tmp[0].data,[np.shape(tco22)[0],1,1])
            co2_tco2_const = carbchem_cube.carbchem(1,mdi,sst,sss,tco22,talk)
            #spg_pco2_tco2_const.append(area_average(co2_tco2_const).data)
            #homogenise_talk
            if j == 0:
                talk_tmp = talk.copy()
            talk2 = talk.copy()
            talk2.data = np.tile(talk_tmp[0].data,[np.shape(talk2)[0],1,1])
            co2_talk_const = carbchem_cube.carbchem(1,mdi,sst,sss,tco2,talk2)
            #spg_pco2_talk_const.append(area_average(co2_talk_const).data)
            coord = cube.coord('time')
            dt = coord.units.num2date(coord.points)
            year = np.array([coord.units.num2date(value).year for value in coord.points])
            #run_year.append(year)
            temp_array0[sizing[j]:sizing[j+1]] = time_mean1.data
            temp_array1[sizing[j]:sizing[j+1]] = time_mean2.data
            temp_array2[sizing[j]:sizing[j+1]] = area_average(co2).data
            temp_array3[sizing[j]:sizing[j+1]] = area_average(co2_sst_const).data
            temp_array4[sizing[j]:sizing[j+1]] = area_average(co2_sss_const).data
            temp_array5[sizing[j]:sizing[j+1]] = area_average(co2_tco2_const).data
            temp_array6[sizing[j]:sizing[j+1]] = area_average(co2_talk_const).data
            temp_array7[sizing[j]:sizing[j+1]] = year 
        spg_fgco2.append(temp_array0)
        stg_fgco2.append(temp_array1)
        spg_pco2.append(temp_array2)
        spg_pco2_sst_const.append(temp_array3)
        spg_pco2_sss_const.append(temp_array4)
        spg_pco2_tco2_const.append(temp_array5)
        spg_pco2_talk_const.append(temp_array6)
        run_year.append(temp_array7)


for i,dummy in enumerate(all_names[0,:]):
    print i
    temp_array0 = np.zeros(91+61+96)
    for j,run_tmp in enumerate(all_names[:,i]):
        #if i >= 114:
        #run_name = run.split('/')[7]
        #run_names.append(run_name)
        tmp = glob.glob(directory+'/'+run_tmp)
        if np.size(tmp) > 0:
            run = tmp[0]
            cube = load_and_agregate_cube(run,'02','00','200')
            time_mean1 = area_average(extract_region(cube))
            temp_array0[sizing[j]:sizing[j+1]] = time_mean1.data
        atm_co2.append(temp_array0)


for i,dummy in enumerate(all_names[0,:]):
    print i
    temp_array0 = np.zeros(91+61+96)
    temp_array1 = np.zeros(91+61+96)
    temp_array2 = np.zeros(91+61+96)
    temp_array3 = np.zeros(91+61+96)
    for j,run_tmp in enumerate(all_names[:,i]):
        #if i >= 114:
        #run_name = run.split('/')[7]
        #run_names.append(run_name)
        tmp = glob.glob(directory+'/'+run_tmp)
        if np.size(tmp) > 0:
            run = tmp[0]
            tco2 = extract_region(load_and_agregate_cube(run,'02','00','103')/(1026.*1000.)*1.0e3)
            time_mean = area_average(tco2)
            temp_array0[sizing[j]:sizing[j+1]] = time_mean.data
            talk = extract_region(load_and_agregate_cube(run,'02','00','104')/(1026.*1000.)*1.0e3)
            time_mean = area_average(talk)
            temp_array1[sizing[j]:sizing[j+1]] = time_mean.data
            sst = extract_region(load_and_agregate_cube(run,'02','00','101'))
            time_mean = area_average(sst)
            temp_array2[sizing[j]:sizing[j+1]] = time_mean.data
            sss = extract_region(load_and_agregate_cube(run,'02','00','102')*1000+35.0)
            time_mean = area_average(sss)
            temp_array3[sizing[j]:sizing[j+1]] = time_mean.data
        spg_tco2.append(temp_array0)
        spg_talk.append(temp_array1)
        spg_sst.append(temp_array2)
        spg_sss.append(temp_array3)
 


#import pickle
#with open('/home/ph290/Documents/python_scripts/pickles/density_plot.pickle', 'w') as f:
#    pickle.dump([run_year,spg_pco2_talk_const,spg_pco2_tco2_const,spg_pco2_sss_const,spg_pco2_sst_const,spg_pco2,stg_fgco2,spg_fgco2,all_names,directory,atm_co2,spg_tco2,spg_talk,spg_sst,spg_sss], f)

'''

'''

#[run_names,spg_fgco2,stg_fgco2,spg_pco2,spg_pco2_sst_const,spg_pco2_sss_const,spg_pco2_tco2_const,spg_pco2_talk_const,run_year] = pickle.load(f)


with open('/home/ph290/Documents/python_scripts/pickles/density_plot.pickle', 'r') as f:
    [run_year,spg_pco2_talk_const,spg_pco2_tco2_const,spg_pco2_sss_const,spg_pco2_sst_const,spg_pco2,stg_fgco2,spg_fgco2,all_names,directory,atm_co2,spg_tco2,spg_talk,spg_sst,spg_sss] = pickle.load(f)

'''

'''

from scipy.stats import gaussian_kde

limit_years = 30.0

plt.close('all')
plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = low_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_talk_const[i]
    x = low_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x,y,c=z,s=50,edgecolor = '')

plt.title('Low-frequency variability (30 year low-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$ calculated with constant alkalinity')
plt.ylabel('Normalised Modelled Subpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_alk_low_pass.ps')
#plt.show(block = True)


plt.close('all')
plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = low_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_sst_const[i]
    x = low_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    plt.scatter(x,y)

plt.title('Low-frequency variability (30 year low-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$ calculated with constant SST')
plt.ylabel('Normalised Modelled Subpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_sst_low_pass.ps')
#plt.show(block = True)


plt.close('all')
plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = low_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_sss_const[i]
    x = low_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    plt.scatter(x,y)

plt.title('Low-frequency variability (30 year low-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$ calculated with constant salinity')
plt.ylabel('Normalised Modelled Subpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_sss_low_pass.ps')
#plt.show(block = False)

plt.close('all')
plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = low_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_tco2_const[i]
    x = low_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    plt.scatter(x,y)

plt.title('Low-frequency variability (30 year low-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$ calculated with constant DIC')
plt.ylabel('Normalised Modelled Subpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_tco2_low_pass.ps')
#plt.show(block = False)


limit_years = 5.0

plt.close('all')
plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = high_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_talk_const[i]
    x = high_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    plt.scatter(x,y)

plt.title('high-frequency variability (5 year high-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$ calculated with constant alkalinity')
plt.ylabel('Normalised Modelled Subpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_alk_high_pass.ps')
#plt.show(block = False)


plt.close('all')
plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = high_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_sst_const[i]
    x = high_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    plt.scatter(x,y)

plt.title('high-frequency variability (5 year high-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$ calculated with constant SST')
plt.ylabel('Normalised Modelled Subpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_sst_high_pass.ps')
#plt.show(block = False)


plt.close('all')
plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = high_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_sss_const[i]
    x = high_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    plt.scatter(x,y)

plt.title('high-frequency variability (5 year high-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$ calculated with constant salinity')
plt.ylabel('Normalised Modelled Subpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_sss_high_pass.ps')
#plt.show(block = False)

plt.close('all')
plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = high_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_tco2_const[i]
    x = high_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    plt.scatter(x,y)

plt.title('high-frequency variability (5 year high-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$ calculated with constant DIC')
plt.ylabel('Normalised Modelled Subpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_tco2_high_pass.ps')
#plt.show(block = False)

'''

'''
#4 on a plot
'''

#with open('/home/ph290/Documents/python_scripts/pickles/qump_co2sb.pickle', 'w') as f:
#    pickle.dump([run_names,spg_fgco2,stg_fgco2,spg_pco2,spg_pco2_sst_const,spg_pco2_sss_const,spg_pco2_tco2_const,spg_pco2_talk_const,run_year,atm_co2], f)

with open('/home/ph290/Documents/python_scripts/pickles/qump_co2sb.pickle', 'r') as f:
    [run_names,spg_fgco2,stg_fgco2,spg_pco2,spg_pco2_sst_const,spg_pco2_sss_const,spg_pco2_tco2_const,spg_pco2_talk_const,run_year,atm_co2] = pickle.load(f)



font = {'family' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)


limit_years = 5.0

plt.close('all')

plt.subplot(2, 2, 1)

xb = []
yb = []
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = high_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_talk_const[i]
    x = high_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    x.mask = x.mask | y.mask
    y.mask = x.mask | y.mask
    x = x[np.logical_not(x.mask)]
    y = y[np.logical_not(y.mask)]
    xb.extend(x)
    yb.extend(y)
	


xb = np.array(xb)
yb = np.array(yb)
xy = np.vstack([xb,yb]) 
# xy = np.ma.masked_invalid(xy)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')
#     plt.scatter(x,y)

plt.title('High-frequency variability\n(5 year high-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$\ncalculated with constant alkalinity')
plt.ylabel('Normalised ESM\nSubpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_alk_high_pass.ps')
#plt.show(block = False)

plt.subplot(2, 2, 2)

xb = []
yb = []
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = high_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_sst_const[i]
    x = high_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    x.mask = x.mask | y.mask
    y.mask = x.mask | y.mask
    x = x[np.logical_not(x.mask)]
    y = y[np.logical_not(y.mask)]
    xb.extend(x)
    yb.extend(y)
	


xb = np.array(xb)
yb = np.array(yb)
xy = np.vstack([xb,yb]) 
# xy = np.ma.masked_invalid(xy)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')


plt.title('High-frequency variability\n(5 year high-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$\ncalculated with constant SST')
plt.ylabel('Normalised ESM\nSubpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_sst_high_pass.ps')
#plt.show(block = False)

plt.subplot(2, 2, 3)

xb = []
yb = []
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = high_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_sss_const[i]
    x = high_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    x.mask = x.mask | y.mask
    y.mask = x.mask | y.mask
    x = x[np.logical_not(x.mask)]
    y = y[np.logical_not(y.mask)]
    xb.extend(x)
    yb.extend(y)
	


xb = np.array(xb)
yb = np.array(yb)
xy = np.vstack([xb,yb]) 
# xy = np.ma.masked_invalid(xy)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')


plt.title('High-frequency variability\n(5 year high-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$\ncalculated with constant salinity')
plt.ylabel('Normalised ESM\nSubpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_sss_high_pass.ps')
#plt.show(block = False)

plt.subplot(2, 2, 4)

xb = []
yb = []
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = high_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_tco2_const[i]
    x = high_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    x.mask = x.mask | y.mask
    y.mask = x.mask | y.mask
    x = x[np.logical_not(x.mask)]
    y = y[np.logical_not(y.mask)]
    xb.extend(x)
    yb.extend(y)
	


xb = np.array(xb)
yb = np.array(yb)
xy = np.vstack([xb,yb]) 
# xy = np.ma.masked_invalid(xy)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')


plt.title('High-frequency variability\n(5 year high-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$\ncalculated with constant DIC')
plt.ylabel('Normalised ESM\nSubpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)

plt.tight_layout()
#plt.show()
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/high_pass.ps')



'''
#4 on a plot low-pass
'''

font = {'family' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)


limit_years = 30.0

plt.close('all')

plt.subplot(2, 2, 1)
xb = []
yb = []
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = low_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_talk_const[i]
    x = low_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    x.mask = x.mask | y.mask
    y.mask = x.mask | y.mask
    x = x[np.logical_not(x.mask)]
    y = y[np.logical_not(y.mask)]
    xb.extend(x)
    yb.extend(y)
	


xb = np.array(xb)
yb = np.array(yb)
xy = np.vstack([xb,yb]) 
# xy = np.ma.masked_invalid(xy)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')

plt.title('Low-frequency variability\n(5 year low-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$\ncalculated with constant alkalinity')
plt.ylabel('Normalised ESM\nSubpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_alk_low_pass.ps')
#plt.show(block = False)

plt.subplot(2, 2, 2)

xb = []
yb = []
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = low_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_sst_const[i]
    x = low_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    x.mask = x.mask | y.mask
    y.mask = x.mask | y.mask
    x = x[np.logical_not(x.mask)]
    y = y[np.logical_not(y.mask)]
    xb.extend(x)
    yb.extend(y)
	


xb = np.array(xb)
yb = np.array(yb)
xy = np.vstack([xb,yb]) 
# xy = np.ma.masked_invalid(xy)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')


plt.title('Low-frequency variability\n(5 year low-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$\ncalculated with constant SST')
plt.ylabel('Normalised ESM\nSubpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_sst_low_pass.ps')
#plt.show(block = False)

plt.subplot(2, 2, 3)

xb = []
yb = []
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = low_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_sss_const[i]
    x = low_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    x.mask = x.mask | y.mask
    y.mask = x.mask | y.mask
    x = x[np.logical_not(x.mask)]
    y = y[np.logical_not(y.mask)]
    xb.extend(x)
    yb.extend(y)
	


xb = np.array(xb)
yb = np.array(yb)
xy = np.vstack([xb,yb]) 
# xy = np.ma.masked_invalid(xy)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')


plt.title('Low-frequency variability\n(5 year low-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$\ncalculated with constant salinity')
plt.ylabel('Normalised ESM\nSubpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)
#plt.savefig('/home/ph290/Documents/figures/n_atl_paper/const_sss_low_pass.ps')
#plt.show(block = False)

plt.subplot(2, 2, 4)

xb = []
yb = []
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = low_pass_filter(y,limit_years)
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))
    y = y/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_pco2_tco2_const[i]
    x = low_pass_filter(x,limit_years)
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))
    x = x/np.max(x)
    x.mask = x.mask | y.mask
    y.mask = x.mask | y.mask
    x = x[np.logical_not(x.mask)]
    y = y[np.logical_not(y.mask)]
    xb.extend(x)
    yb.extend(y)
	


xb = np.array(xb)
yb = np.array(yb)
xy = np.vstack([xb,yb]) 
# xy = np.ma.masked_invalid(xy)
z = gaussian_kde(xy)(xy)
idx = z.argsort()
xb, yb, z = xb[idx], yb[idx], z[idx]
plt.scatter(xb,yb,c=z,s=20,edgecolor = '')


plt.title('Low-frequency variability\n(5 year low-pass filter)')
plt.xlabel('Normalised Subpolar pCO$_2$\ncalculated with constant DIC')
plt.ylabel('Normalised ESM\nSubpolar pCO$_2$')
plt.xlim(0,1)
plt.ylim(0,1)

plt.tight_layout()
#plt.show()
plt.savefig('/home/ph290/Documents/figures/n_atl_paper/low_pass.ps')





'''










plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = np.ma.masked_where(((y > 500) | (y < -500)),y)
    y = (y-np.min(y))/np.max(y)
    y = np.ma.masked_where(y > 500,y)
    x = spg_sss[i]
    x = np.ma.masked_where(((x > 500) | (x < -500)),x)
    x = (x-np.min(x))/np.max(x)
    plt.scatter(x,y)


#plt.ylim(-50,300)
#plt.xlim(1850,2100)
plt.show(block = False)



plt.figure(0)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i]
    y = y - np.mean(y[0:20])
    y = np.ma.masked_where(y > 500,y)
    plt.plot(run_year[i],y)

plt.ylim(-50,300)
plt.xlim(1850,2100)
plt.show(block = False)

for i in range(np.shape(run_year)[0]):
    plt.plot(run_year[i],spg_fgco2[i]-np.mean(spg_fgco2[i][0:20]))

plt.ylim(-2,8)
plt.xlim(1850,2100)
plt.show()


for i in range(np.shape(run_year)[0]):
    plt.plot(run_year[i],stg_fgco2[i]-np.mean(stg_fgco2[i][0:20]))

plt.ylim(-2,2)
plt.xlim(1850,2100)
plt.show()


plt.figure(1)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i] - spg_pco2_talk_const[i]
    y = y - np.mean(y[0:20])
    y = np.ma.masked_where(y > 500,y)
    plt.plot(run_year[i],y)

plt.ylim(-50,300)
plt.xlim(1850,2100)
plt.show(block = False)


plt.figure(2)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i] - spg_pco2_tco2_const[i]
    y = y - np.mean(y[0:20])
    y = np.ma.masked_where(y > 500,y)
    plt.plot(run_year[i],y)

plt.ylim(-50,300)
plt.xlim(1850,2100)
plt.show(block = False)



plt.figure(3)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i] - spg_pco2_sst_const[i]
    y = y - np.mean(y[0:20])
    y = np.ma.masked_where(y > 500,y)
    plt.plot(run_year[i],y)

plt.ylim(-50,300)
plt.xlim(1850,2100)
plt.show(block = False)



plt.figure(4)
for i in range(np.shape(run_year)[0]):
    y = spg_pco2[i] - spg_pco2_sss_const[i]
    y = y - np.mean(y[0:20])
    y = np.ma.masked_where(y > 500,y)
    plt.plot(run_year[i],y)

plt.ylim(-50,300)
plt.xlim(1850,2100)
plt.show(block = False)







alpha_val = 0.2
lw = 2
smoothing = 20



consolidated_lines = []
consolidated_years = []
consolidated_lines_spg = []


for j in range(np.size(all_names[0,:])):
    loc = []
    data2 = []
    data2_spg = []
    years2 = []
    for i,name in enumerate(all_names[:,0]):
        tmp_loc = np.where(np.array(run_names) == all_names[i,j])[0]
        if np.size(tmp_loc) > 0:
            loc = np.append(loc,tmp_loc)
    for l in loc:
        minr = np.min(run_date[int(l)])
        maxr = np.max(run_date[int(l)])
        if maxr == 1949:
            data2 = np.append(data2,run_global_means[int(l)][1:])
            data2_spg = np.append(data2_spg,run_global_means_spg[int(l)][1:])
            years2 = np.append(years2,run_date[int(l)][1:])
        if maxr == 2004:
            data2 = np.append(data2,run_global_means[int(l)][6:])
            data2_spg = np.append(data2_spg,run_global_means_spg[int(l)][6:])
            years2 = np.append(years2,run_date[int(l)][6:])
        if maxr == 2099:
            data2 = np.append(data2,run_global_means[int(l)][1:])
            data2_spg = np.append(data2_spg,run_global_means_spg[int(l)][1:])
            years2 = np.append(years2,run_date[int(l)][1:])
    consolidated_lines.append(data2)
    consolidated_lines_spg.append(data2_spg-np.mean(data2_spg[0:20]))
    consolidated_years.append(years2)

plt.close('all')
#fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig = plt.figure(figsize= (12,6))
fig.add_subplot(121) 
for i,data in enumerate(consolidated_lines):
    if np.size(data) > 0:
        if np.size(np.where(data <= -0.5)) == 0:
            if np.size(np.where(data[1::]-data[0:-1] > 0.5)) == 0:
                plt.plot(consolidated_years[i],data,'k',alpha = alpha_val,linewidth=lw)
                plt.xlabel('year')
                plt.ylabel('air-sea CO$_2$ flux anomaly (mol-C m$^{-2}$ yr$^{-1}$)')
                plt.title('Global flux')

for i,data in enumerate(consolidated_lines):
    if np.size(data) > 0:
        if np.size(np.where(data <= -0.5)) == 0:
            if np.size(np.where(data[1::]-data[0:-1] > 0.5)) == 0:
                plt.plot(consolidated_years[i],run_mean.running_mean(data-np.mean(data[0:20]),smoothing),alpha = 0.5,linewidth=lw)

fig.add_subplot(122)
for i,data in enumerate(consolidated_lines_spg):
    if np.size(data) > 0:
        #if np.size(np.where(data <= -0.5)) == 0:
        #if np.size(np.where(data[1::]-data[0:-1] > 0.5)) == 0:
        plt.plot(consolidated_years[i],data,'k',alpha = alpha_val,linewidth=lw)
        plt.xlabel('year')
        plt.title('Subpolar flux')
        #plt.ylabel('Subpolar Gyre air-sea CO$_2$ flux anomaly (mol-C m$^{-2}$ yr$^{-1}$)')

for i,data in enumerate(consolidated_lines_spg):
    #if np.size(data) > 0:
    #    if np.size(np.where(data <= -0.5)) == 0:
    #        if np.size(np.where(data[1::]-data[0:-1] > 0.5)) == 0:
                plt.plot(consolidated_years[i],run_mean.running_mean(data,smoothing),alpha = 0.5,linewidth=lw)

mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/n_atl/figure7_aug20.png')



'''
#map
'''

brewer_cmap = mpl_cm.get_cmap('RdBu_r')
brewer_cmap2 = mpl_cm.get_cmap('Reds')

consolidated_maps = []
run_maps = np.array(run_maps)

for j in range(np.size(all_names[0,:])):
    loc = []
    for i,name in enumerate(all_names[:,0]):
        tmp_loc = np.where(np.array(run_names) == all_names[i,j])[0]
        if np.size(tmp_loc) > 0:
            loc = np.append(loc,tmp_loc)
    if np.size(loc) == 3:
        tmp = run_maps[np.array(map(int,loc))]
        out = tmp[0].copy()
        out.data = data1 = tmp[0].data + tmp[1].data + tmp[2].data
        consolidated_maps.append(out)



mean_map = consolidated_maps[0].copy()
std_map = consolidated_maps[0].copy()

all_data = np.zeros([np.size(consolidated_maps),np.size(run_maps[0].data[:,0]),np.size(run_maps[0].data[0,:])])
all_data[:] = np.NAN

all_data2 = all_data.copy()

for i in np.arange(np.size(consolidated_maps)):
    all_data[i,:,:] = np.ma.masked_invalid(consolidated_maps[i].data)

all_data_mean = np.mean(all_data,axis = 0)

for i in np.arange(np.size(consolidated_maps)):
    all_data2[i,:,:] = all_data[i,:,:]

all_data_stdev = np.std(all_data2,axis = 0)

mean_map.data = np.ma.masked_array(all_data_mean)
mean_map.data[np.where(mean_map.data == 0)] = np.NAN
mean_map.data = np.ma.masked_invalid(mean_map.data)

std_map.data = np.ma.masked_array(all_data_stdev)
std_map.data[np.where(std_map.data == 0)] = np.NAN
std_map.data = np.ma.masked_invalid(std_map.data)

plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(211)
cs = iplt.contourf(mean_map,np.linspace(-2000,2000,51),cmap = brewer_cmap)
plt.gca().coastlines()
cbar = plt.colorbar(cs, ticks=[-1800, -900,0,900, 1800],orientation='horizontal', shrink=.6, pad=0.1, aspect=10)
cbar.ax.set_xlabel('Mean cumulative CO$_2$ uptake (mol-C m$^{-2}$)')

ax1 = fig.add_subplot(212)
zmin = 0
zmax = 300
std_map.data[np.where(std_map.data >= zmax)] = zmax
cs = iplt.contourf(std_map,np.linspace(zmin,zmax,51),cmap = brewer_cmap2)
plt.gca().coastlines()
cbar = plt.colorbar(cs, ticks=[0, 100,200,300],orientation='horizontal', shrink=.6, pad=0.1, aspect=10)
cbar.ax.set_xlabel('Inter-model standard deviation in \ncumulative CO$_2$ uptake (mol-C m$^{-2}$)')

plt.tight_layout()
plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/n_atl/figurex_aug20.png')


'''
