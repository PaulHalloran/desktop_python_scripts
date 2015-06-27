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

def model_names2(directory):
	models = []
	files = glob.glob(directory+'/*.nc')
	for file in files:
		models.append(file.split('/')[-1].split('_')[0])
	return np.array(np.unique(models))
    
def model_names(directory):
    files = glob.glob(directory+'/*.nc')
    models_tmp = []
    exp_tmp = []
    models2 = []
    for file in files:
        statinfo = os.stat(file)
        if statinfo.st_size >= 1:
            exp_tmp.append(file.split('/')[-1].split('_')[2])
            models_tmp.append(file.split('/')[-1].split('_')[0])
    exps = np.unique(exp_tmp)
    models = np.unique(models_tmp)
    exp_tmp2 = []
    for model in models:
        files = glob.glob(directory+'/'+model+'*.nc')
        for file in files:
            statinfo = os.stat(file)
            if statinfo.st_size >= 1:
                exp_tmp2.append(file.split('/')[-1].split('_')[2])
            test = np.size(np.unique(exp_tmp))
        if test == 2:
            models2.append(model)
    return np.array(models2)


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

plt.figure()
qplt.contourf(cube2.collapsed('TMNTH',iris.analysis.MEAN, weights = grid_areas),np.linspace(-100,100))
plt.gca().coastlines()
plt.show(block = False)

plt.figure()
qplt.contourf(cube.collapsed('TMNTH',iris.analysis.MEAN, weights = grid_areas),np.linspace(300,500))
plt.gca().coastlines()
plt.show(block = False)

ts = cube2.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights = grid_areas)
plt.figure()
qplt.plot(ts)
plt.show()

plt.figure()
qplt.pcolormesh(cube2[-20])
plt.gca().coastlines()
plt.show(block = False)

plt.figure()
qplt.pcolormesh(cube2[-1])
plt.gca().coastlines()
plt.show(block = False)

west = -70
east = -10
south = 50
north = 70
spg_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
spg_cube = cube2.extract(spg_region)

west = -70
east = -10
south = -10
north = 30
sbtrp_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
sbtrp_cube = cube2.extract(sbtrp_region)

grid_areas1 = iris.analysis.cartography.area_weights(spg_cube)
ts1 = spg_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights = grid_areas1)
grid_areas2 = iris.analysis.cartography.area_weights(sbtrp_cube)
ts2 = sbtrp_cube.collapsed(['latitude','longitude'],iris.analysis.MEAN, weights = grid_areas2)

plt.figure()
qplt.plot(ts2-ts1,'blue')
#qplt.plot(ts2,'red')
plt.show(block = False)

plt.figure()
qplt.plot(ts1,'blue')
qplt.plot(ts2,'red')
plt.show(block = False)

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

plt.figure()
plt.scatter(spg_yr,spg_data,color = 'blue')
l1 = plt.plot(sbtrp_yr,results.params[1]*spg_yr+results.params[0],'blue', label='subpolar (50-70N)')
plt.scatter(sbtrp_yr,sbtrp_data,color = 'red')
l2 = plt.plot(sbtrp_yr,results2.params[1]*sbtrp_yr+results2.params[0],'red', label='subtropical (0-30N)')
plt.legend()
plt.ylabel('SOCAT anomaly from atm. CO2')
plt.xlabel('year')
plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/socat_analysis.png')
#plt.show(block = False)


'''
model analysis
'''

#model_data = '/media/usb_external1/cmip5/spco2/'
model_data = '/data/temp/ph290/andy_w_analysis/processed_annual'

models = model_names2(model_data)
experiments = ['historical','rcp85']

model_spg_cube1_mean_acc = []
model_sbtrp_cube1_mean_acc = []
model_spg_cube2_mean_acc = []
model_sbtrp_cube2_mean_acc = []

models_final = []

for i,model in enumerate(models):
    print 'processing model '+np.str(i+1)
    cube1 = iris.load_cube(model_data+model+'_*'+experiments[0]+'*.nc')/0.101325
    test1 = cube1[0].collapsed(['latitude','longitude'],iris.analysis.MEAN).data
    if (test1 <= 0.002):
        cube1 = iris.load_cube(model_data+model+'_*'+experiments[0]+'*.nc')/0.000000101325
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
        south = -10
        north = 30
        sbtrp_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
        model_sbtrp_cube1 = cube1.extract(sbtrp_region)
        grid_areas = iris.analysis.cartography.area_weights(model_spg_cube1)
        model_spg_cube1_mean = model_spg_cube1.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
        grid_areas = iris.analysis.cartography.area_weights(model_sbtrp_cube1)
        model_sbtrp_cube1_mean = model_sbtrp_cube1.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
        model_spg_cube1_mean_acc.append(model_spg_cube1_mean)
        model_sbtrp_cube1_mean_acc.append(model_sbtrp_cube1_mean)

        cube2 = iris.load_cube(model_data+model+'_*'+experiments[1]+'*.nc')/0.101325
        if (test1 <= 0.002):
            cube2 = iris.load_cube(model_data+model+'_*'+experiments[1]+'*.nc')/0.000000101325
        iris.coord_categorisation.add_year(cube2, 'time', name='year')
        cube2 = cube2.aggregated_by('year', iris.analysis.MEAN)
        new_cube = []
        for cube_slice in cube2.slices(['latitude','longitude']):
            loc = np.where(atm_co2[:,0] == cube_slice.coord('year').points)
            cube_slice -= atm_co2[loc[0],1]
            new_cube.append(cube_slice)
        cube2 = iris.cube.CubeList(new_cube)
        cube2 = iris.cube.CubeList.merge(cube2)[0]
        if not cube2.coord('latitude').has_bounds():
            cube2.coord('latitude').guess_bounds()
        if not cube2.coord('longitude').has_bounds():
            cube2.coord('longitude').guess_bounds()
        west = 360-70
        east = 360-10
        south = 50
        north = 70
        spg_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
        model_spg_cube2 = cube2.extract(spg_region)
        west = 360-70
        east = 360-10
        south = -10
        north = 30
        sbtrp_region = iris.Constraint(longitude=lambda v: west <= v <= east,latitude=lambda v: south <= v <= north)
        model_sbtrp_cube2 = cube2.extract(sbtrp_region)
        grid_areas = iris.analysis.cartography.area_weights(model_spg_cube2)
        model_spg_cube2_mean = model_spg_cube2.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
        grid_areas = iris.analysis.cartography.area_weights(model_sbtrp_cube2)
        model_sbtrp_cube2_mean = model_sbtrp_cube2.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights = grid_areas)
        model_spg_cube2_mean_acc.append(model_spg_cube2_mean)
        model_sbtrp_cube2_mean_acc.append(model_sbtrp_cube2_mean)


plt.figure()
for i,model in enumerate(models_final):
    if not model == 'MRI-ESM1':
        try:
            qplt.plot(model_spg_cube1_mean_acc[i],'blue')
            qplt.plot(model_sbtrp_cube1_mean_acc[i],'red')
            qplt.plot(model_spg_cube2_mean_acc[i],'blue')
            qplt.plot(model_sbtrp_cube2_mean_acc[i],'red')
        except:
            print 'model: '+model+' failed'

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.show(block = False)



colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k')

'''
NOTE - I'm just processing this more neaty for andy's analysis - i.e. no gap between hist and future.
This will be in /data/temp/ph290/andy_w_analysis/processed
'''


plt.figure()
for i,model in enumerate(models_final):
    if not model == 'MRI-ESM1':
        try:
            yrs = model_spg_cube1_mean_acc[i].coord('year').points
            plt.plot(yrs,running_mean.running_mean(model_spg_cube1_mean_acc[i].data,10),linewidth = 2,  color=colors[i],label=model)
            yrs = model_spg_cube2_mean_acc[i].coord('year').points
            plt.plot(yrs,running_mean.running_mean(model_spg_cube2_mean_acc[i].data,10),linewidth = 2,  color=colors[i])
        except:
            print 'model: '+model+' failed'

plt.legend(loc = 3,prop={'size':8})
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.xlim([1860,2100])
plt.title('SPG air-sea difference 10yr running mean')
plt.savefig('/home/ph290/Documents/figures/n_atl_paper_II/cmip5_spg_smoothed.png')
#plt.show(block = False)
