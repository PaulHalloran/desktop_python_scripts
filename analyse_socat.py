
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
import iris.coord_categorisation
import iris.analysis
import iris.analysis.cartography
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
import running_mean
import glob
from matplotlib.lines import Line2D
import scipy.stats
import matplotlib as mpl



def model_names(directory):
	models = []
	files = glob.glob(directory+'/*.nc')
	for file in files:
		models.append(file.split('/')[-1].split('_')[0])
	return np.array(np.unique(models))
	
font = {'family' : 'monospace',
        'weight' : 'bold',
        'family' : 'serif',
        'size'   : 14}
    

cube = iris.load_cube('/data/data1/ph290/observations/SOCAT_tracks_gridded_monthly_v2.nc','fCO2 mean - per cruise weighted')
iris.coord_categorisation.add_year(cube, 'TMNTH', name='year')
cube = cube.aggregated_by('year', iris.analysis.MEAN)

atm_co2 = np.genfromtxt('/data/data1/ph290/observations/historical_and_rcp85_atm_co2.txt',skip_header = 1)

new_cube = []

#calculate the differences from the average atm. CO2 for that year
for cube_slice in cube.slices(['latitude','longitude']):
    loc = np.where(atm_co2[:,0] == cube_slice.coord('year').points)
    cube_slice -= atm_co2[loc[0],1]
    new_cube.append(cube_slice)

cube2 = iris.cube.CubeList(new_cube)
cube2 = iris.cube.CubeList.merge(cube2)[0]

if not cube2.coord('latitude').has_bounds():
    cube2.coord('latitude').guess_bounds()
if not cube2.coord('longitude').has_bounds():
    cube2.coord('longitude').guess_bounds()

grid_areas = iris.analysis.cartography.area_weights(cube2)

# plt.figure()
# qplt.contourf(cube2.collapsed('TMNTH',iris.analysis.MEAN, weights = grid_areas),np.linspace(-100,100))
# plt.gca().coastlines()
# plt.show(block = False)
# 
# plt.figure()
# qplt.contourf(cube.collapsed('TMNTH',iris.analysis.MEAN, weights = grid_areas),np.linspace(300,500))
# plt.gca().coastlines()
# plt.show(block = False)
# 
# ts = cube2.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights = grid_areas)
# plt.figure()
# qplt.plot(ts)
# plt.show()
# 
# plt.figure()
# qplt.pcolormesh(cube2[-20])
# plt.gca().coastlines()
# plt.show(block = False)
# 
# plt.figure()
# qplt.pcolormesh(cube2[-1])
# plt.gca().coastlines()
# plt.show(block = False)

west = -70
east = -10
south = 50
north = 70
spg_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
spg_cube = cube2.extract(spg_region)

west = -70
east = -10
south = 0
north = 30
sbtrp_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
sbtrp_cube = cube2.extract(sbtrp_region)

grid_areas1 = iris.analysis.cartography.area_weights(spg_cube)
ts1 = spg_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights = grid_areas1)
grid_areas2 = iris.analysis.cartography.area_weights(sbtrp_cube)
ts2 = sbtrp_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights = grid_areas2)
# 
# plt.figure()
# qplt.plot(ts2-ts1,'blue')
# #qplt.plot(ts2,'red')
# plt.show(block = False)
# 
# plt.figure()
# qplt.plot(ts1,'blue')
# qplt.plot(ts2,'red')
# plt.show(block = False)

loc = np.where(ts1.coord('year').points >= 2001)
loc2 = ts1.coord('TMNTH').points[loc[0][0]]
ts1b = ts1.extract(iris.Constraint(TMNTH = lambda time_tmp: time_tmp >= loc2))

spg_data = ts1b.data
spg_yr = ts1b.coord('year').points
X2 = sm.add_constant(spg_yr)
model = sm.OLS(spg_data,X2)
results = model.fit()
results.params

loc = np.where(ts2.coord('year').points >= 2001)
loc2 = ts2.coord('TMNTH').points[loc[0][0]]
ts2b = ts2.extract(iris.Constraint(TMNTH = lambda time_tmp: time_tmp >= loc2))

sbtrp_data = ts2b.data
sbtrp_yr = ts2b.coord('year').points
X2b = sm.add_constant(sbtrp_yr)
model2 = sm.OLS(sbtrp_data,X2b)
results2 = model2.fit()
results2.params


# plt.figure()
# plt.scatter(spg_yr,spg_data,color = 'blue')
# l1 = plt.plot(sbtrp_yr,results.params[1]*spg_yr+results.params[0],'blue', label='subpolar (50-70N)')
# plt.scatter(sbtrp_yr,sbtrp_data,color = 'red')
# l2 = plt.plot(sbtrp_yr,results2.params[1]*sbtrp_yr+results2.params[0],'red', label='subtropical (0-30N)')
# plt.legend()
# plt.ylabel('SOCAT anomaly from atm. CO2')
# plt.xlabel('year')
# plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/socat_analysis.png')
# #plt.show(block = False)


'''
#model analysis
'''

model_data = '/data/temp/ph290/andy_w_analysis/processed_annual/'

models = model_names(model_data)

model_spg_cube1_mean_acc = []
model_sbtrp_cube1_mean_acc = []

models_final = []

for i,model in enumerate(models):
    print 'processing model '+np.str(i+1)
    cube1 = iris.load_cube(model_data+model+'_*.nc')/0.101325
    test1 = cube1[0].collapsed(['latitude','longitude'],iris.analysis.MEAN).data
    if (test1 <= 0.002):
        cube1 = iris.load_cube(model_data+model+'_*.nc')/0.000000101325
    test = cube1[0].collapsed(['latitude','longitude'],iris.analysis.MEAN).data
    if (test >= 200) & (test <= 2000):
        models_final.append(model)
        iris.coord_categorisation.add_year(cube1, 'time', name='year')
        cube1 = cube1.aggregated_by('year', iris.analysis.MEAN)
        new_cube = []
        for cube_slice in cube1.slices(['latitude','longitude']):
            loc = np.where(atm_co2[:,0] == cube_slice.coord('year').points)
            cube_slice -= atm_co2[loc[0],1]
            new_cube.append(cube_slice)
        cube1 = iris.cube.CubeList(new_cube)
        cube1 = iris.cube.CubeList.merge(cube1)[0]
        if not cube1.coord('latitude').has_bounds():
            cube1.coord('latitude').guess_bounds()
        if not cube1.coord('longitude').has_bounds():
            cube1.coord('longitude').guess_bounds()
        west = 360-70
        east = 360-10
        south = 50
        north = 70
        spg_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
        model_spg_cube1 = cube1.extract(spg_region)
        west = 360-70
        east = 360-10
        south = 0
        north = 30
        sbtrp_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
        model_sbtrp_cube1 = cube1.extract(sbtrp_region)
        grid_areas = iris.analysis.cartography.area_weights(model_spg_cube1)
        model_spg_cube1_mean = model_spg_cube1.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
        grid_areas = iris.analysis.cartography.area_weights(model_sbtrp_cube1)
        model_sbtrp_cube1_mean = model_sbtrp_cube1.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
        model_spg_cube1_mean_acc.append(model_spg_cube1_mean)
        model_sbtrp_cube1_mean_acc.append(model_sbtrp_cube1_mean)


# plt.figure()
# for i,model in enumerate(models_final):
#     if not model == 'MRI-ESM1':
#         try:
#             qplt.plot(model_spg_cube1_mean_acc[i],'blue')
#             qplt.plot(model_sbtrp_cube1_mean_acc[i],'red')
#         except:
#             print 'model: '+model+' failed'
# 
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
# plt.show(block = False)



colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k')

colors = ('b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b')

linestyles = ['-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':']
linestyles = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']

markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass



'''
#NOTE - I'm just processing this more neatly for andy's analysis - i.e. no gap between hist and future.
#This will be in /data/temp/ph290/andy_w_analysis/processed
'''



font = {'family' : 'monospace',
        'weight' : 'bold',
        'family' : 'serif',
        'size'   : 14}

mpl.rcdefaults()
mpl.rc('font', **font)

font_size = 14
font_weight = 'bold'

plt.close('all')
plt.figure()
ax1 = plt.subplot(1,1,1)
ax1.fill_between([2000,2012,2012,2000], [-100,-100,100,160], y2=0, where=None,alpha= 0.25,color='gray')
for i,model in enumerate(models_final):
    if not model == 'MRI-ESM1':
        try:
            yrs = model_spg_cube1_mean_acc[i].coord('year').points
            #plt.plot(yrs,running_mean.running_mean(model_spg_cube1_mean_acc[i].data,5),linewidth = 5,linestyle=linestyles[i],  color=colors[i],label=model,alpha= 0.5)
            y = running_mean.running_mean(model_spg_cube1_mean_acc[i].data,10)
            colour = 'g'
            if min(y[150:-31]) < min(y[-30:-1]): colour = 'b'
            if min(y[0:150]) < min(y[151:-1]): colour = 'r'
            ax1.plot(yrs,(y-scipy.stats.nanmean(y[0:20]))*-1.0,linewidth = 6,linestyle=linestyles[i],  color=colour,label=model,alpha= 0.4)
        except:
            print 'model: '+model+' failed'


results_box = np.genfromtxt('/home/ph290/box_modelling/boxmodel_6_box_back_to_basics/results/rcp85_spg_box_model_qump_results_3.csv',delimiter = ',')
ax1.plot(results_box[:,0],(results_box[:,1]-results_box[0,1])*(44/12.0),'k',linewidth = 6,alpha= 0.4,linestyle = '-')
print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
print 'Am I right in changing the units here????? I think I need to convert all of the box model stuff - or all o fthe other stuff to teh different units.......!!!!!!!!!!!!!!!'
print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'


plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))

#plt.title('SPG air-sea difference 20yr running mean')
ax1.set_ylabel('SPG ocean minus atm. [CO$_2$] (ppm)\nanomaly from preindustrial', multialignment='center',fontweight = font_weight,fontsize = font_size)
ax1.set_xlabel('Year',fontweight = font_weight,fontsize = font_size)

legend2 = ax1.legend(loc = 3,prop={'size':10, 'family' : 'normal','weight' : 'bold'},ncol = 3).draw_frame(False)

ax2 = ax1.twinx()
b1 = ax2.plot([1,2],'r',linewidth = 6,alpha= 0.5) 
b2 = ax2.plot([1,2],'b',linewidth = 6,alpha= 0.5) 
b3 = ax2.plot([1,2],'g',linewidth = 6,alpha= 0.5) 
legend1 = ax2.legend([b1[0], b2[0], b3[0]], ['peak before 2013','peak 2013-2020','no peak before 2070'],loc = 2,prop={'size':10, 'family' : 'normal','weight' : 'bold'},ncol = 1).draw_frame(False)
frame1 = plt.gca()
for xlabel_i in frame1.axes.get_xticklabels():
    xlabel_i.set_visible(False)
    xlabel_i.set_fontsize(0.0)
for xlabel_i in frame1.axes.get_yticklabels():
    xlabel_i.set_fontsize(0.0)
    xlabel_i.set_visible(False)
for tick in frame1.axes.get_xticklines():
    tick.set_visible(False)
for tick in frame1.axes.get_yticklines():
    tick.set_visible(False)


ax1.set_xlim([1860,2100])
ax1.set_ylim([-40,60])
ax2.set_xlim([1860,2100])
ax2.set_ylim([-40,60])

#plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/cmip5_spg_smoothed2.png')
plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/cmip5_spg_smoothed2.pdf')
#plt.show(block = False)
#plt.close('all')


'''
Socat plot
'''



mpl.rcdefaults()
mpl.rc('font', **font)

marker_size = 30

#plt.figure(figsize=(6, 8))

plt.figure()
plt.scatter(spg_yr,spg_data,color = 'blue',s=marker_size)
plt.plot(sbtrp_yr,results.params[1]*spg_yr+results.params[0],'blue',linewidth = 3)
plt.scatter(sbtrp_yr,sbtrp_data,color = 'red',s=marker_size)
plt.plot(sbtrp_yr,results2.params[1]*sbtrp_yr+results2.params[0],'red',linewidth = 3)
plt.ylabel('ocean [CO$_2$] minus atm. [CO$_2$] (ppm)',fontweight = font_weight,fontsize = font_size)
plt.xlabel('year',fontweight = font_weight,fontsize = font_size)

for i,model in enumerate(models_final):
    if not model == 'MRI-ESM1':
        try:
            yrs = model_spg_cube1_mean_acc[i].coord('year').points
            y = running_mean.running_mean(model_spg_cube1_mean_acc[i].data,10)
            y = model_spg_cube1_mean_acc[i].data
            plt.plot(yrs,y,linewidth = 6, color='b',alpha= 0.1)
        except:
            print 'model: '+model+' failed'
            
for i,model in enumerate(models_final):
    if not model == 'MRI-ESM1':
        try:
            yrs = model_sbtrp_cube1_mean_acc[i].coord('year').points
            y = running_mean.running_mean(model_sbtrp_cube1_mean_acc[i].data,10)
            y = model_sbtrp_cube1_mean_acc[i].data
            plt.plot(yrs,y,linewidth = 6, color='r',alpha= 0.1)
        except:
            print 'model: '+model+' failed'

plt.xlim([2000,2012])
plt.ylim([40,-90])

b1 = plt.scatter([1,2],[1,2],color='k',s=marker_size) 
b2 = plt.plot([1,2],'k',linewidth = 3) 
b3 = plt.plot([1,2],'k',linewidth = 6,alpha= 0.1)
b4 = plt.plot([1,2],'b',linewidth = 6,alpha= 0.5) 
b5 = plt.plot([1,2],'r',linewidth = 6,alpha= 0.5) 
plt.legend([b1, b2[0], b3[0], b4[0], b5[0]], ['SOCAT annual mean', 'SOCAT linear trend','CMIP5','Subpolar (50-70N)','Subtropical (0-30N)'],prop={'size':12, 'family' : 'normal','weight' : 'bold'},ncol = 2).draw_frame(False)

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
plt.tight_layout()
plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/socat_analysis.png')
#plt.show(block = False)

