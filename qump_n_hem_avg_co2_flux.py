

import iris
import numpy as np
import matplotlib.pyplot as plt
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
run_global_means = []
run_global_means_spg = []

run_date = []
run_maps = []

lon_west = 0
lon_east = 355
lat_south = 0
lat_north = 90 

for i,run in enumerate(runs):
    #if i >= 114:
        print i
        run_name = run.split('/')[7]
        run_names.append(run_name)
        cube = iris.load_cube(run+'/*02.30.249*.pp',iris.AttributeConstraint(STASH='m02s30i249'),callback=my_callback)
        iris.coord_categorisation.add_year(cube, 'time', name='year2')
        cube2 = cube.aggregated_by('year2', iris.analysis.MEAN)
        run_maps.append(cube2.collapsed('time', iris.analysis.SUM))
        cube2.coord('longitude').guess_bounds()
        cube2.coord('latitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube2)
        time_mean = cube2.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
        run_global_means.append(time_mean.data)
        cube3 = cube2.intersection(longitude=(lon_west, lon_east))
        cube3 = cube3.intersection(latitude=(lat_south, lat_north))
        try:
            cube3.coord('longitude').guess_bounds()
            cube3.coord('latitude').guess_bounds()
        except:
            None
        grid_areas3 = iris.analysis.cartography.area_weights(cube3)
        time_mean3 = cube3.collapsed(['longitude', 'latitude'], iris.analysis.SUM, weights=grid_areas3)
        run_global_means_spg.append(time_mean3.data)
        coord = cube2.coord('time')
        year = np.array([coord.units.num2date(value).year for value in coord.points])
        run_date.append(year)
        np.savetxt(output_directory + run_name + '_n_hem.txt',np.vstack((year,time_mean3.data)).T,delimiter=',')


import iris
import numpy as np
import matplotlib.pyplot as plt
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

output_directory = ('/home/ph290/data1/qump_co2/global_avg/')

files = glob.glob(output_directory+'*_n_hem.txt')

years = []
run_global_means_n_hem = []

for file in files:
	tmp = np.genfromtxt(file,delimiter = ',')
	years.append(tmp[:,0])
	run_global_means_n_hem.append(tmp[:,1])

fig = plt.figure()
for i,data in enumerate(run_global_means_n_hem):
	plt.plot(years[i],run_global_means_n_hem[i]*12.0/1.0e15)

plt.xlabel('year')
plt.ylabel('N. Hem. ocean CO2 uptake Pg-C/yr')
plt.show()

'''


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
                plt.plot(consolidated_years[i],data/12.0/1.0e15,'k',alpha = alpha_val,linewidth=lw)
                plt.xlabel('year')
                plt.ylabel('air-sea CO$_2$ flux anomaly (Pg C yr$^{-1}$)')
                plt.title('global flux')

for i,data in enumerate(consolidated_lines):
    if np.size(data) > 0:
        if np.size(np.where(data <= -0.5)) == 0:
            if np.size(np.where(data[1::]-data[0:-1] > 0.5)) == 0:
                plt.plot(consolidated_years[i],run_mean.running_mean(data/12.0/1.0e15-(np.mean(data[0:20]/12.0/1.0e15)),smoothing),alpha = 0.5,linewidth=lw)

fig.add_subplot(122)
for i,data in enumerate(consolidated_lines_spg):
    if np.size(data) > 0:
        #if np.size(np.where(data <= -0.5)) == 0:
        #if np.size(np.where(data[1::]-data[0:-1] > 0.5)) == 0:
        plt.plot(consolidated_years[i],data/12.0/1.0e15,'k',alpha = alpha_val,linewidth=lw)
        plt.xlabel('year')
        plt.title('Subpolar flux')
        #plt.ylabel('N. Hem. air-sea CO$_2$ flux anomaly (mol-C m$^{-2}$ yr$^{-1}$)')

for i,data in enumerate(consolidated_lines_spg):
    #if np.size(data) > 0:
    #    if np.size(np.where(data <= -0.5)) == 0:
    #        if np.size(np.where(data[1::]-data[0:-1] > 0.5)) == 0:
                plt.plot(consolidated_years[i],run_mean.running_mean(data/12.0/1.0e15,smoothing),alpha = 0.5,linewidth=lw)

mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
#plt.show(block = False)
plt.savefig('/home/ph290/Documents/figures/n_atl/figure_n_hem.png')


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
