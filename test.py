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


temporary_cube = mmm_density_cube.intersection(longitude = (west, east))
density_cube_mm_mean_reg = temporary_cube.intersection(latitude = (south, north))

density_cube_mm_mean_reg.coord('latitude').guess_bounds()
density_cube_mm_mean_reg.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(density_cube_mm_mean_reg)
density_ts = density_cube_mm_mean_reg.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
try:
	iris.coord_categorisation.add_year(density_ts, 'time', name='year2')
except:
	print 'year2 already exists'
	

y2 = density_ts.data
y2 = y2/(np.max(21)-np.min(21))
y2 = y2 -np.mean(21)

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

