import iris
import glob
import subprocess
import iris.analysis.cartography
import numpy as np
import matplotlib.pyplot as plt
import iris.quickplot as qplt

directory = '/home/ph290/data1/aerosol_data/'
run = ['historical/','rcp26/','rcp45/','rcp60/','rcp85/']

# for j in range(np.size(run)):
#     files = glob.glob(directory+run[j]+'*.nc')
#     #note that we only need to do this once - it is just easier to do it from python rather than on the commend line individuallt...
#     for file in files:
#         subprocess.Popen("ncatted -O -a bounds,time,d,, "+file,shell=True)

# run_data=[]
# run_year=[]
# run_month=[]

# for j in range(np.size(run)):
# #for j in range(1):
#     files = glob.glob(directory+run[j]+'*.nc')
#     years = []
#     for file in files:
#         tmp1=file.split('_')
#         tmp2=tmp1[-1].split('-')
#         years.append(tmp2[0])

#     years=np.array(years)
#     years_sorted=np.sort(years)

#     data=[]
#     for i,file_name in enumerate(files):
#         loc=np.where(years == years_sorted[i])
#         print files[loc[0]]
#         tmp = iris.load(files[loc[0]],'SO4 concentration')[0]
#         tmp.coord('latitude').guess_bounds()
#         tmp.coord('longitude').guess_bounds()
#         grid_areas = iris.analysis.cartography.area_weights(tmp)
#         data.append(tmp.collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=grid_areas))

#     shape = np.shape(data[0])
#     tmp_data=np.zeros(np.size(data))
#     yr=np.zeros(np.size(data))
#     yr.fill(np.nan)
#     for k in range(np.size(data)):
#         tmp_data[k] = np.sum(data[k][:].data)
#         tmp_time = data[k][0].coord('time')
#         if tmp_time.points > 0:
#             dt = tmp_time.units.num2date(tmp_time.points)
#             yr[k] = dt[0].year

#     run_data.append(tmp_data)
#     run_year.append(yr) 




names = ['Historical','RCP 2.6','RCP 4.5','RCP 6.0','RCP 8.5']

# lns=[]
# for i in range(np.size(run)):
#         ln=plt.plot(run_year[i],run_data[i],linewidth=3,label=names[i])
#         lns += ln
# 
# plt.xlabel('year')
# plt.ylabel('SO$_4$ concentration')
# 
# labs=[l.get_label() for l in lns]
# plt.legend(lns,labs,loc=0).draw_frame(False)
# plt.tight_layout()
# plt.show()


'''
spatial maps (and timeseries)
'''

variables = ['OC1 concentration','OC2 concentration','SO4 concentration','CB1 concentration','CB2 concentration']
variables_names = ['Organic carbon concentration (hydrophobic)','Organic carbon concentration (hydrophillic)','Sulphate concentration','Black carbon concentration (hydrophobic)','Black carbon concentration (hydrophobic)']
var_max = [2.8e-10,7e-10,2.0e-9,2.0e-10,2.0e-10]


var_data = []                   
for k,dummy in enumerate(variables):
	run_data = []
	for j,dummy in enumerate(run):
		files = glob.glob(directory+run[j]+'*.nc')
		years = []
		for file in files:
			tmp1=file.split('_')
			tmp2=tmp1[-1].split('-')
			years.append(tmp2[0])

		years=np.array(years)
		years_sorted=np.sort(years)

		data = []	
		for i,file_name in enumerate(files):
			loc=np.where(years == years_sorted[i])
			print files[loc[0]]
			tmp = iris.load(files[loc[0]],variables[k])[0]
			tmp.coord('latitude').guess_bounds()
			tmp.coord('longitude').guess_bounds()
			x = tmp.collapsed(['atmosphere_hybrid_sigma_pressure_coordinate','time'], iris.analysis.MEAN)
			grid_areas = iris.analysis.cartography.area_weights(tmp)
			data_tmp = tmp.collapsed(['longitude','latitude','atmosphere_hybrid_sigma_pressure_coordinate','time'], iris.analysis.MEAN,weights=grid_areas)
			data.append(data_tmp.data)

			plt.figure()
			qplt.contourf(x,np.linspace(0.0,var_max[k]))
			plt.title(names[j]+' year '+years_sorted[i]+' '+variables_names[k])
			plt.gca().coastlines()
			tmp_names = names[j].replace(' ','')
			tmp_names = tmp_names.replace('.','')
			tmp_variable = variables[k].replace(' ','')
			if np.logical_not(years_sorted[i] == '1950'):
				plt.savefig('/home/ph290/Documents/teaching/masters/aerosol_conc_maps/'+tmp_names+'_'+years_sorted[i]+'_'+tmp_variable+'.png')

		run_data.append(data)

	# var_data.append(run_data)

        # plt.figure()
	# lns=[]
	# for i in range(np.size(run)):
	# 		ln=plt.plot(run_year[i],run_data[i],linewidth=3,label=names[i])
	# 		lns += ln

	# plt.xlabel('year')
	# plt.ylabel('concentration')
	# plt.title(variables_names[k])
	# labs=[l.get_label() for l in lns]
	# plt.legend(lns,labs,loc=0).draw_frame(False)
	# plt.tight_layout()
	# plt.savefig('/home/ph290/Documents/teaching/masters/aerosol_conc_timeseries/'+tmp_variable+'.png')

for k,dummy in enumerate(variables):
	for j,dummy in enumerate(run):
		tmp_names = names[j].replace(' ','')
		tmp_names = tmp_names.replace('.','')
		tmp_variable = variables[k].replace(' ','')
		subprocess.Popen('convert -delay 50 /home/ph290/Documents/teaching/masters/aerosol_conc_maps/'+tmp_names+'*'+tmp_variable+'.png /home/ph290/Documents/teaching/masters/aerosol_conc_maps/'+tmp_names+tmp_variable+'.gif',shell=True)
		
		
		
