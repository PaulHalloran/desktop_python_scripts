'''
NOTE! launch python2.7 with 'sudo' first
'''
# import iris
# import iris.analysis.cartography
# import iris.analysis
# import numpy as np
# import matplotlib.pyplot as plt
# import iris.plot as iplt
# import iris.coords
# import iris.quickplot as qplt
# import cartopy.crs as ccrs
# import shelve
# import iris.coord_categorisation
# from scipy import signal
# import os
# import glob
# 
# 
# def movingaverage(interval, window_size):
#     window= np.ones(int(window_size))/float(window_size)
#     return np.convolve(interval, window, 'same')
# 
# 
# variable_index=1
# 
# cefas_dir='/Users/ph290/Public/temp_data/cefas_data/'
# cefas_files=['Cefas_hindcast_monthlyvals_site_a.nc','Cefas_hindcast_monthlyvals_site_b.nc','Cefas_hindcast_monthlyvals_site_c.nc','Cefas_hindcast_monthlyvals_site_d.nc']
# 
# moll_dir='/Users/ph290/Public/temp_data/butler_data/'
# moll_files=['Chickens_IOM_Glycymeris_chronology_1941.txt','Ramsey_IOM_Glycymeris_chronology_1921.txt','IOM_Arctica_chronology_1516.txt','Tiree_Passage_Scotland_Glycymeris_Chronology_1805.txt','north_sea_arctica_f1_chron.txt']
# moll_site_names=['Isle of Man 1','Isle of Man 2','Isle of Man 3','Tiree Passage (nr Oban)','N. Sea (N)']
# 
# # Isle of Man Arctica from several positions off the west coast of the IOM, but centred around 4 50 E  54 10 N.   Between 25 and 80 metres depth.  For the full range see the map in Butler et al 2009 (EPSL).
# # Isle of Man Glycymeris: Chickens is 54 06.031N, 04 23.195W  about 60 metres depth                                   
# #                                         Ramsey is 54  06.031N, 04 23.195W about 35-40 metres depth  (These are from the attached paper by Brocas et al)
# # North Sea F1 Arctica      59 23.1N, 0 31.0E  about 140m depth (from Butler et al 2009 (Palaeoceanography))
# # Tiree Passage Glycymeris  56 37N, 6 24W, 50m water depth   (unpublished)
# 
# 
# cube0=iris.load_raw(cefas_dir+cefas_files[0])
# cube1=iris.load_raw(cefas_dir+cefas_files[1])
# cube2=iris.load_raw(cefas_dir+cefas_files[2])
# cube3=iris.load_raw(cefas_dir+cefas_files[3])
# cubes=[cube0,cube1,cube2,cube3]
# 
# cube=cube0
# 
# var_name=[]
# for i in np.arange(np.size(cube)):
#     var_name.append(cube[i].metadata[1])
# 
# coord = cube[0].coord('time')
# dt = coord.units.num2date(coord.points)
# cfas_year = np.array([coord.units.num2date(value).year for value in coord.points])
# cfas_month = np.array([coord.units.num2date(value).month for value in coord.points])
# cfas_time=cfas_year+cfas_month/12.0
# 
# 
# 
# 
# data=np.genfromtxt(moll_dir+moll_files[4])
# mollusk_year=data[:,0]
# mollusk_gr=data[:,1]
# 
# month=3
# 
# tmp=cefas_data=cubes[0].extract(var_name[0])[0].data
# monthly_loc=np.where(cfas_month == 3)
# monthly_cefas_data=cefas_data[monthly_loc]
# temp_data=np.zeros([monthly_cefas_data.size,17])
# 
# fig = plt.figure()
# for i in np.arange(np.size(cube)):
#     ax1 = fig.add_subplot(4,5,i+1)
#     for j in np.arange(np.size(cefas_files)):
#         cefas_data=cubes[j].extract(var_name[i])[0].data
#         monthly_loc=np.where(cfas_month == month)
#         monthly_cefas_data=cefas_data[monthly_loc]
#         temp_data[:,j]=np.array(monthly_cefas_data)
#         unique_years=np.unique(cfas_year)
#         ax1.plot(unique_years,movingaverage(monthly_cefas_data,1))
# 
#     #ax2 = ax1.twinx()
#     #ax2.plot(mollusk_year,movingaverage(mollusk_gr,10), 'r')
#     #plt.xlim(1950,2010)
#     #plt.title(var_name[i])
# 
# plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.2)
# plt.show()
# print 'month = '+str(month)
# 
# '''
# '''
# 
# mollusk_year=[]
# mollusk_gr=[]
# for i in np.arange(np.size(moll_files)):
#     data=np.genfromtxt(moll_dir+moll_files[i])
#     mollusk_year.append(data[:,0])
#     mollusk_gr.append(data[:,1])
# 
# fig = plt.figure()
# i=6
# 
# j=3
# 
# ax1 = fig.add_subplot(1,1,1)
# cube_tmp=cubes[j].extract(var_name[i])
# iris.coord_categorisation.add_year(cube_tmp[0], 'time', name='year2')
# cube_tmp2 = cube_tmp[0].aggregated_by('year2', iris.analysis.MEAN)
# cefas_data=cube_tmp2.data
# unique_years=np.unique(cfas_year)
# ln1=ax1.plot(unique_years,movingaverage(cefas_data,1),label = var_name[i],linewidth=2.0)
# ax1.set_ylabel('Suspension feeder')
# 
# k=4
# ax2 = ax1.twinx()
# y1=movingaverage(mollusk_gr[4][:],10)
# y1=y1[5:-5]
# ln2=ax2.plot(mollusk_year[4][5:-5],y1, 'r',label = '10yr smoothed mollusk growth rate',linewidth=2.0)
# ax2.set_ylabel('growth rate')
# plt.xlim(1950,2010)
# lns=ln1+ln2
# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs).draw_frame(False)
# ax1.set_xlabel('year')
# 
# 
# plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.2)
# plt.show()
# 
# '''
# '''
# 
# i=16
# month=[3]
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# #for j in np.arange(np.size(cefas_files)):
#     # cefas_data=cubes[j].extract(var_name[i])[0].data
#     # monthly_loc=np.where(cfas_month == month)
#     # monthly_cefas_data=cefas_data[monthly_loc]
#     # unique_years=np.unique(cfas_year)
#     # ax1.plot(unique_years,movingaverage(monthly_cefas_data,1), 'b')
# 
# cefas_data=cubes[3].extract(var_name[i])[0].data
# monthly_loc1=np.where(cfas_month == month[0])
# #monthly_loc2=np.where(cfas_month == month[1])
# #monthly_loc3=np.where(cfas_month == month[2])
# monthly_cefas_data1=cefas_data[monthly_loc1]
# #monthly_cefas_data2=cefas_data[monthly_loc2]
# #monthly_cefas_data3=cefas_data[monthly_loc3]
# monthly_cefas_data=np.mean([monthly_cefas_data1],axis=0)
# unique_years=np.unique(cfas_year)
# 
# ax1.plot(unique_years,monthly_cefas_data, 'b')
# ax1.plot([1987,1987],[-1000,1000], 'g')
# #plt.ylim(260,282.5)
# plt.ylim(245,315)
# 
# ax2 = ax1.twinx()
# mv_avg=12
# mov_avg_data=movingaverage(mollusk_gr[4][:],mv_avg)
# ax2.plot(mollusk_year[4][mv_avg/2.0:-1*(mv_avg/2.0)],mov_avg_data[mv_avg/2.0:-1*(mv_avg/2.0)], 'r')
# ax2.plot(mollusk_year[4][:],mollusk_gr[4][:], 'grey')
# plt.ylim(0.0,2.25)
# plt.xlim(1860,2010)
# ax1.set_ylabel('BFM sea bed oxygen')
# ax1.yaxis.label.set_color('blue')
# ax2.set_ylabel(' arctica growth rate (10 year moving box smoothed)')
# ax2.yaxis.label.set_color('red')
# ax1.set_xlabel('year')
# plt.show()
# 
# 
# '''
# '''
# 
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# for j in np.arange(np.size(cefas_files)):
#     cefas_data=cubes[j].extract(var_name[i])[0].data
#     monthly_loc=np.where(cfas_month == month)
#     monthly_cefas_data=cefas_data[monthly_loc]
#     unique_years=np.unique(cfas_year)
#     ax1.plot(unique_years,movingaverage(monthly_cefas_data,1), 'b')
# 
# plt.show()
# 
# mollusk_year=[]
# mollusk_gr=[]
# for i in np.arange(np.size(moll_files)):
#     data=np.genfromtxt(moll_dir+moll_files[i])
#     mollusk_year.append(data[:,0])
#     mollusk_gr.append(data[:,1])
# 
# 
# 
# fig=plt.figure()
# for i in np.arange(np.size(moll_files)):
#     fig.add_subplot(2,3,i+1)
#     plt.plot(mollusk_year[i][:],mollusk_gr[i][:])
#     plt.title(moll_site_names[i])
# 
# plt.show()
# 
# '''
# '''
# 
# i=2
# moving_avg_val=10
# data_tmp=movingaverage(mollusk_gr[i][:],moving_avg_val)
# data_tmp=data_tmp[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
# yrs_tmp=mollusk_year[i][:]
# yrs_tmp=yrs_tmp[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
# plt.plot(yrs_tmp,data_tmp)
# plt.title(moll_site_names[i])
# 
# plt.show()
# 
# '''
# AMO-like Oban dataset
# '''
# 
# amo_box = iris.Constraint(longitude=lambda v: -75.0 <= v <= -7.5,latitude=lambda v: 0 <= v <= 60)
# off_scotland_box = iris.Constraint(longitude=lambda v: -15 <= v <= 0,latitude=lambda v: 50 <= v <= 65)
# 
# hadsst_dir='/Users/ph290/Public/temp_data/misc_data/'
# #HadSST.3.1.0.0.anomalies.*.nc'
# os.chdir(hadsst_dir)
# 
# all_ssts_amo=np.zeros([100,164])
# all_ssts_scotland=np.zeros([100,164])
# 
# #Read in all SST ensemble members
# for i,file in enumerate(glob.glob("HadSST.3.1.0.0.anomalies.*.nc")):
#     print file
#     sst_cubes=iris.load(hadsst_dir+file)
#     sst_cube=sst_cubes[1]
#     #AMO box
#     amo_sst=sst_cube.extract(amo_box)  
#     #mean from monthly data to annual data
#     iris.coord_categorisation.add_year(amo_sst, 'time', name='year')
#     amo_sst_cube_annually_meaned = amo_sst.aggregated_by('year', iris.analysis.MEAN)
#     #area average
#     amo_sst_cube_annually_meaned.coord('latitude').guess_bounds()
#     amo_sst_cube_annually_meaned.coord('longitude').guess_bounds()
#     grid_areas = iris.analysis.cartography.area_weights(amo_sst_cube_annually_meaned)
#     amo_sst_area_avg = amo_sst_cube_annually_meaned.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
#     all_ssts_amo[i,:]=amo_sst_area_avg.data
#     #Scotland box
#     scot_sst=sst_cube.extract(off_scotland_box)  
#     #mean from monthly data to annual data
#     iris.coord_categorisation.add_year(scot_sst, 'time', name='year')
#     scot_sst_cube_annually_meaned = scot_sst.aggregated_by('year', iris.analysis.MEAN)
#     #area average
#     scot_sst_cube_annually_meaned.coord('latitude').guess_bounds()
#     scot_sst_cube_annually_meaned.coord('longitude').guess_bounds()
#     grid_areas = iris.analysis.cartography.area_weights(scot_sst_cube_annually_meaned)
#     scot_sst_area_avg = scot_sst_cube_annually_meaned.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
#     all_ssts_scotland[i,:]=scot_sst_area_avg.data
# 
# 
# #read in HadSST data
# #sst_file='/Users/ph290/Public/temp_data/misc_data/HadISST_sst.nc'
# sst_file='/Users/ph290/Public/temp_data/misc_data/HadSST.3.1.0.0.median.nc'
# sst_cubes=iris.load(sst_file)
# sst_cube=sst_cubes[1]
# 
# #extract the AMO box
# amo_box = iris.Constraint(longitude=lambda v: -75.0 <= v <= -7.5,latitude=lambda v: 0 <= v <= 60)
# amo_sst=sst_cube.extract(amo_box)
# 
# off_scotland_box = iris.Constraint(longitude=lambda v: -15 <= v <= 0,latitude=lambda v: 50 <= v <= 65)
# off_scotland_sst=sst_cube.extract(off_scotland_box)
# #qplt.contourf(off_scotland_sst[0],100)
# #plt.show()
# 
# '''
# amo data
# '''
# 
# #mean from monthly data to annual data
# iris.coord_categorisation.add_year(amo_sst, 'time', name='year')
# amo_sst_cube_annually_meaned = amo_sst.aggregated_by('year', iris.analysis.MEAN)
# 
# #area average
# amo_sst_cube_annually_meaned.coord('latitude').guess_bounds()
# amo_sst_cube_annually_meaned.coord('longitude').guess_bounds()
# grid_areas = iris.analysis.cartography.area_weights(amo_sst_cube_annually_meaned)
# amo_sst_area_avg = amo_sst_cube_annually_meaned.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
# 
# '''
# off scotland data
# '''
# 
# #mean from monthly data to annual data
# iris.coord_categorisation.add_year(off_scotland_sst, 'time', name='year')
# off_scotland_sst_cube_annually_meaned = off_scotland_sst.aggregated_by('year', iris.analysis.MEAN)
# 
# #area average
# off_scotland_sst_cube_annually_meaned.coord('latitude').guess_bounds()
# off_scotland_sst_cube_annually_meaned.coord('longitude').guess_bounds()
# grid_areas = iris.analysis.cartography.area_weights(off_scotland_sst_cube_annually_meaned)
# off_scotland_sst_area_avg = off_scotland_sst_cube_annually_meaned.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
# 
# '''
# '''
# 
# coord = amo_sst_cube_annually_meaned.coord('time')
# dt = coord.units.num2date(coord.points)
# hadsst_year = np.array([coord.units.num2date(value).year for value in coord.points])
# 
# '''
# plotting
# '''
# 
# fig=plt.figure()
# i=3
# ax1 = fig.add_subplot(2,1,1)
# ax1.set_xlim(1800,2020)
# moving_avg_val=2
# 
# 
# for j,dummy in enumerate(glob.glob("HadSST.3.1.0.0.anomalies.*.nc")):
#     data_tmp4=movingaverage(all_ssts_scotland[j,1:-1],moving_avg_val)
#     data_tmp4=data_tmp4[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
#     yrs_tmp4=hadsst_year[1:-1]
#     yrs_tmp4=yrs_tmp4[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
#     ln1=ax1.plot(yrs_tmp4,data_tmp4,'#7A7A7A',linewidth=2.0,label='hadsst3 local ssts (3yr smoothed)')
#     #ln1=ax1.plot(hadsst_year[1:-1],all_ssts[j,1:-1],'#7A7A7A',linewidth=2.0,label='hadsst3 AMO (linear detrended)')
# 
# #signal.detrend()
# 
# ax1.set_ylim(-0.8,1.0)
# #plt.title(moll_site_names[i])
# ax1.set_xlabel('year')
# 
# ax2 = ax1.twinx()
# ax2.set_xlim(1800,2020)
# 
# data_tmp2=movingaverage(mollusk_gr[i][:],moving_avg_val)
# data_tmp2=data_tmp2[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
# yrs_tmp2=mollusk_year[i][:]
# yrs_tmp2=yrs_tmp2[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
# ln2=ax2.plot(yrs_tmp2,data_tmp2,color='r',linewidth=3,label='Tiree Passage (nr Oban) growth rates (3yr smoothed)')
# #ln2=ax2.plot(mollusk_year[i][:],mollusk_gr[i][:],color='r',linewidth=3,label='Tiree Passage (nr Oban) growth rates')
# ax2.set_ylim(0.5,1.5)
# 
# lns = ln1+ln2
# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs).draw_frame(False)
# 
# ax3 = fig.add_subplot(2,1,2)
# ax3.set_xlim(1800,2020)
# plt.xlabel('year')
# for j,dummy in enumerate(glob.glob("HadSST.3.1.0.0.anomalies.*.nc")):
#     data_tmp4=movingaverage(all_ssts_scotland[j,1:-1],moving_avg_val)
#     data_tmp4=data_tmp4[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
#     yrs_tmp4=hadsst_year[1:-1]
#     yrs_tmp4=yrs_tmp4[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
#     ln3=ax3.plot(yrs_tmp4,signal.detrend(data_tmp4),'#7A7A7A',linewidth=2.0,label='hadsst3 local ssts, detrended (3yr smoothed)')
# 
# for j,dummy in enumerate(glob.glob("HadSST.3.1.0.0.anomalies.*.nc")):
#     data_tmp4=movingaverage(all_ssts_amo[j,1:-1],moving_avg_val)
#     data_tmp4=data_tmp4[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
#     yrs_tmp4=hadsst_year[1:-1]
#     yrs_tmp4=yrs_tmp4[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
#     ln4=ax3.plot(yrs_tmp4,signal.detrend(data_tmp4),'k',linewidth=2.0,label='hadsst3 AMO index (3yr smoothed)')
# 
# y1=signal.detrend(data_tmp4)
# ax3.fill_between(yrs_tmp4, 0, y1, where=y1>=0, facecolor='red', interpolate=True)
# ax3.fill_between(yrs_tmp4, 0, y1, where=0>=y1, facecolor='blue', interpolate=True)
# 
# lns = ln3+ln4
# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs).draw_frame(False)
# 
# #plt.title('Atlantic Multidecadal Oscillation Index')
# 
# #signal.detrend()
# 
# 
# plt.show()
# #NOTE HADSST3 has missing data - it is uninterpolated. As is HadCRUT4
# 
# '''
# next bit is just un-smoothed
# '''
# 
# # fig=plt.figure()
# # i=3
# # ax1 = fig.add_subplot(1,1,1)
# # data_tmp2=mollusk_gr[i][:]
# # yrs_tmp2=mollusk_year[i][:]
# # ax1.plot(yrs_tmp2,data_tmp2,linewidth=3)
# # #plt.ylim(0.4,2.0)
# # ax1.title(moll_site_names[i])
# 
# # ax2 = ax1.twinx()
# 
# # data_tmp4=amo_sst_area_avg.data
# # yrs_tmp4=hadsst_year
# # ax2.plot(yrs_tmp4,signal.detrend(data_tmp4),'r',linewidth=3)
# # #note - this is linear detrended
# 
# # ax3= ax2.twinx()
# 
# # data_tmp3=movingaverage(off_scotland_sst_area_avg.data[0:-1],moving_avg_val)
# # data_tmp3=data_tmp3[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
# # yrs_tmp3=hadsst_year[0:-1]
# # yrs_tmp3=yrs_tmp3[moving_avg_val/2.0:-1*(moving_avg_val/2.0)]
# # ax3.plot(yrs_tmp3,data_tmp3,'k',linewidth=3)
# 
# 
# 
# '''
# MLR
# '''
# 
# nao_data=np.genfromtxt('/Users/ph290/Public/temp_data/misc_data/nao.dat')
# winter_nao=nao_data[3:,[1,2,3,12]]
# nao_year=nao_data[3:,0]
# nao_winter_index = np.mean(winter_nao,axis=1)
# 
# month=3
# 
# tmp=cefas_data=cubes[0].extract(var_name[0])[0].data
# monthly_loc=np.where(cfas_month == 3)
# monthly_cefas_data=cefas_data[monthly_loc]
# temp_data=np.zeros([monthly_cefas_data.size,17])
# 
# i=3
# for j in np.arange(np.size(cube)):
#     cefas_data=cubes[i].extract(var_name[j])[0].data
#     monthly_loc=np.where(cfas_month == month)
#     monthly_cefas_data=cefas_data[monthly_loc]
#     temp_data[:,j]=np.array(monthly_cefas_data)
#     unique_years=np.unique(cfas_year)
# 
# 
# # yr2=mollusk_year[91:]
# # y=np.array(mollusk_gr[91:])
# yr2=mollusk_year[4][91:]
# y_tmp=np.array(mollusk_gr[4][91:])
# moving_avg_val=10
# # y2=movingaverage(np.array(mollusk_gr[91:]),moving_avg_val)
# # y=y2[moving_avg_val/2:-moving_avg_val/2]
# y2=movingaverage(np.array(y_tmp),moving_avg_val)
# y=y2[moving_avg_val/2:-moving_avg_val/2]
# yr2=yr2[moving_avg_val/2:-moving_avg_val/2]
# 
# yr1=unique_years[:-5]
# x=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,0:16])
# o2=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,16])
# sst=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,7])
# sss=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,12])
# 
# tmp=np.array([7,12])
# x2=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,tmp])
# 
# tmp=np.array([7,12,13,13])
# x3=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,tmp])
# 
# be = np.linalg.lstsq(x,y)
# be2 = np.linalg.lstsq(x2,y)
# be3 = np.linalg.lstsq(x3,y)
# 
# # model is y = b0.x0 + b1.x1 + b2.x2
# 
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# lns1 = ax1.plot(yr2,y, label = 'arctica growth rate (10yr moving box filtered)')
# ax1.set_ylabel('growth rate')
# 
# 
# b=be[0]
# x=x
# #ax1.plot(yr1[moving_avg_val/2:-moving_avg_val/2],b[0]*x[:,0]+b[1]*x[:,1]+b[2]*x[:,2]+b[3]*x[:,3]+b[4]*x[:,4]+b[5]*x[:,5]+b[6]*x[:,6]+b[7]*x[:,7]+b[8]*x[:,8]+b[9]*x[:,9]+b[10]*x[:,10]+b[11]*x[:,11]+b[12]*x[:,12]+b[13]*x[:,13]+b[14]*x[:,14]+b[15]*x[:,15])
# b=be2[0]
# x=x2
# lns2 = ax1.plot(yr1[moving_avg_val/2:-moving_avg_val/2],b[0]*x[:,0]+b[1]*x[:,1], label = 'MLR on BFM SST and SSS')
# b=be3[0]
# x=x3
# lns3 = ax1.plot(yr1[moving_avg_val/2:-moving_avg_val/2],b[0]*x[:,0]+b[1]*x[:,1]+b[2]*x[:,2]+b[3]*x[:,3],'--', label = 'MLR on BFM SST and SSS and sea-bed amonium')
# 
# ax2 = ax1.twinx()
# lns4 = ax2.plot(yr1[moving_avg_val/2:-moving_avg_val/2],o2,'r', label = 'BFM March sea bottom O2')
# plt.ylim(260,282.5)
# ax2.set_ylabel('O2 conc. (mmol O2/m3)')
# 
# lns = lns1+lns2+lns3+lns4
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)
# 
# 
# #ax3 = ax2.twinx()
# #nao_winter_index2=movingaverage(np.array(nao_winter_index),moving_avg_val)
# #nao_winter_index2b=nao_winter_index2[moving_avg_val/2:-moving_avg_val/2]
# #nao_year2=nao_year[moving_avg_val/2:-moving_avg_val/2]
# #ax3.plot(nao_year2,nao_winter_index2b*-1.0,'y')
# 
# 
# plt.show()
# 
# #can get a lot of teh variability from just 7 and 12
# print var_name[7] +' and '+ var_name[12]
# 
# '''
# script to automatically pick out optimal variables to explain growth rates
# Does this bu following a montecarlo approach, randomly taking the values from any month and any variables supplied by cefas, and throwing them into the multiple linear regression - then compares the RMSE of the MLR based timeseries with the mollusk growth rates, and finally selects the lowest RMSE, and prints out the corresponding values
# '''
# 
# #note - benthos will take a decade or so to spin up, so should really do this again ignoring the first decade
# def rmse(ts1,ts2):
#     #ts1 = predicted value
#     #ts2 = true value
#     arraysize=np.size(ts1)
#     diff_sq=np.square(ts1-ts2)
#     mse=np.sum(diff_sq)*(1.0/arraysize)
#     return np.sqrt(mse)
# 
# 
# temp_data=np.zeros([36,12,np.size(cube)])
# 
# for month in range(12):
#     for j in np.arange(np.size(cube)):
#         cefas_data=cubes[3].extract(var_name[j])[0].data
#         monthly_loc=np.where(cfas_month == month+1)
#         monthly_cefas_data=cefas_data[monthly_loc]
#         temp_data[:,month,j]
#         temp_1=np.array(monthly_cefas_data)
#         temp_data[:,month,j]=temp_1[5:-5-5]
#         #accounts of rremoving values equal to a 10yr smoothing
#         unique_years=np.unique(cfas_year)
# 
# yr1=unique_years[:-5]
# 
# variables_array1=np.empty([temp_data.shape[1],temp_data.shape[2]])
# variables_array2=np.empty([temp_data.shape[1],temp_data.shape[2]])
# for j in np.arange(temp_data.shape[1]):
#     for k in np.arange(temp_data.shape[2]):
#         variables_array1[j,k]=k
#         variables_array2[j,k]=j
# 
# variables_array1b=np.reshape(variables_array1,[17*12])
# variables_array2b=np.reshape(variables_array2,[17*12])
# 
# #=var_name[j]+', month '+str(j+1)
# 
# temp_data2=np.reshape(temp_data,[36,17*12])
# 
# yr2=mollusk_year[4][91:]
# y_tmp=np.array(mollusk_gr[4][91:])
# moving_avg_val=10
# # y2=movingaverage(np.array(mollusk_gr[91:]),moving_avg_val)
# # y=y2[moving_avg_val/2:-moving_avg_val/2]
# y2=movingaverage(np.array(y_tmp),moving_avg_val)
# y=y2[moving_avg_val/2:-moving_avg_val/2]
# 
# rmse_log=[]
# input_var1=[]
# input_var2=[]
# input_var3=[]
# 
# for j in range(1000):
# #for j in range(1000):
#     min_var=0
#     max_var=temp_data2.shape[1]-1
#     temp_var1=np.random.randint(min_var,max_var)
#     temp_var2=np.random.randint(min_var,max_var)
#     temp_var3=np.random.randint(min_var,max_var)
#     input_var1=np.append(input_var1,temp_var1)
#     input_var2=np.append(input_var2,temp_var2)
#     input_var3=np.append(input_var3,temp_var3)  
# 
#     tmp3=np.array([temp_var1,temp_var2,temp_var3])
#     be = np.linalg.lstsq(temp_data2[:,tmp3.astype(int)],y)
#     b=be[0]
#     temp_2=temp_data2[:,0]*0.0
#     for i in np.arange(b.size):
#         temp_2 = np.sum([temp_2,b[i]*temp_data2[:,np.int(tmp3[i])]],axis=0)
#     rmse_tmp=rmse(y,temp_2)
#     rmse_log=np.append(rmse_log,rmse_tmp)
# 
# sorted_rmse=np.sort(rmse_log)
# loc=np.where(rmse_log == sorted_rmse[0])
# print rmse_log[loc]
# tmp3=np.array([input_var1[loc],input_var2[loc],input_var3[loc]])
# tmp3=tmp3[:,0]
# be = np.linalg.lstsq(temp_data2[:,tmp3.astype(int)],y)
# b=be[0]
# temp_2=temp_data2[:,0]*0.0
# for i in np.arange(b.size):
#     temp_2 = np.sum([temp_2,b[i]*temp_data2[:,tmp3[i].astype(int)]],axis=0)
# 
# 
# sorted_rmse=np.sort(rmse_log)
# for i in range(20):
#     loc=np.where(rmse_log == sorted_rmse[i])
#     print '##### '+str(i)+' #####'
#     print rmse_log[loc[0]]
#     print var_name[np.int(variables_array1b[np.int(input_var1[loc[0]])])]+', month '+str(np.int(variables_array2b[np.int(input_var1[loc[0]])]))
#     print var_name[np.int(variables_array1b[np.int(input_var2[loc[0]])])]+', month '+str(np.int(variables_array2b[np.int(input_var2[loc[0]])]))
#     print var_name[np.int(variables_array1b[np.int(input_var3[loc[0]])])]+', month '+str(np.int(variables_array2b[np.int(input_var3[loc[0]])]))
# 
# 
# 
# 
# fig0 = plt.figure()
# ax1 = fig0.add_subplot(1,1,1)
# for i in range(1):
#     loc[0]=np.where(rmse_log == sorted_rmse[4])
#     tmp3=np.array([input_var1[loc[0]],input_var2[loc[0]],input_var3[loc[0]]])
#     tmp3=tmp3[:,0]
#     be = np.linalg.lstsq(temp_data2[:,tmp3.astype(int)],y)
#     b=be[0]
#     temp_2=temp_data2[:,0]*0.0
#     for i in np.arange(b.size):
#         temp_2 = np.sum([temp_2,b[i]*temp_data2[:,tmp3[i].astype(int)]],axis=0)
#     lns1 = ax1.plot(yr2[5:-5],y, label = 'arctica growth rate (10yr moving box filtered)')
#     ax1.set_ylabel('growth rate')
#     ax1.plot(yr1[moving_avg_val/2:-moving_avg_val/2],temp_2,'--')
#     ax2 = ax1.twinx()
#     lns4 = ax2.plot(yr1[moving_avg_val/2:-moving_avg_val/2],o2,'r', label = 'BFM March sea bottom O2')
#     plt.ylim(260,282.5)
#     ax2.set_ylabel('O2 conc. (mmol O2/m3)')
# 
# plt.show()
# 
# # ax2 = fig0.add_subplot(2,2,3)
# # ax2.plot(yr2[5:-5],temp_data2[:,np.int(input_var1[loc[0]])])
# # ax2.set_title(var_name[np.int(variables_array1b[np.int(input_var1[loc[0]])])]+', month '+str(np.int(variables_array2b[np.int(input_var1[loc[0]])])))
# 
# # ax3 = fig0.add_subplot(2,2,3)
# # ax3.plot(yr2[5:-5],temp_data2[:,np.int(input_var2[loc[0]])])
# # ax3.set_title(var_name[np.int(variables_array1b[np.int(input_var2[loc[0]])])]+', month '+str(np.int(variables_array2b[np.int(input_var2[loc[0]])])))
# 
# # ax4 = fig0.add_subplot(2,2,4)
# # ax4.plot(yr2[5:-5],temp_data2[:,np.int(input_var3[loc[0]])])
# # ax4.set_title(var_name[np.int(variables_array1b[np.int(input_var3[loc[0]])])]+', month '+str(np.int(variables_array2b[np.int(input_var3[loc[0]])])))
# 
# # plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.2)
# # plt.show()
# 
# output = shelve.open('/Users/ph290/Public/temp_data/output/my_shelf4.dat')
# output['rmse_log']=rmse_log
# output['input_var1']=input_var1
# output['input_var2']=input_var2
# output['input_var3']=input_var3
# output.close()
# 
# # #and restore with:
# # # input = shelve.open('/Users/ph290/Public/temp_data/output/my_shelf.dat')
# # # rmse_log = input['rmse_log']
# # # input_var1 = input['input_var1']
# # # input_var2 = input['input_var2']
# # # input_var3 = input['input_var3']
# # # input.close()
# 
# 
# # and restore with:
# input = shelve.open('/Users/ph290/Public/temp_data/output/my_shelf.dat')
# rmse_log = input['rmse_log']
# input_var1 = input['input_var1']
# input_var2 = input['input_var2']
# input_var3 = input['input_var3']
# input.close()
# 


# plt.ion()

# yr2=mollusk_year[91:]
# y=np.array(mollusk_gr[91:])
#mollusk_year[4][:]
moll_yr_mv_avg=mollusk_year[4][:]
moll_mv_avg=np.array(mollusk_gr[4][:])
moving_avg_val=10
# y2=movingaverage(np.array(mollusk_gr[91:]),moving_avg_val)
# y=y2[moving_avg_val/2:-moving_avg_val/2]
moll_mv_avg2=movingaverage(np.array(moll_mv_avg),moving_avg_val)
moll_mv_avg3=moll_mv_avg2[moving_avg_val/2:-moving_avg_val/2]
moll_yr_mv_avg2=moll_yr_mv_avg[moving_avg_val/2:-moving_avg_val/2]


fig0 = plt.figure()
ax1 = fig0.add_subplot(1,1,1)

sorted_rmse=np.sort(rmse_log)
loc=np.where(rmse_log == sorted_rmse[0])
tmp3=np.array([input_var1[loc[0]],input_var2[loc[0]],input_var3[loc[0]]])
tmp3=tmp3[:,0]
be = np.linalg.lstsq(temp_data2[:,tmp3.astype(int)],y)
b=be[0]
temp_2=temp_data2[:,0]*0.0
for i in np.arange(b.size):
    temp_2 = np.sum([temp_2,b[i]*temp_data2[:,tmp3[i].astype(int)]],axis=0)

lns1 = ax1.plot(moll_yr_mv_avg2,moll_mv_avg3,'b', label = 'A. icelandica growth rate (10yr moving box filtered)',linewidth=3)
ax1.set_ylabel('growth rate', fontsize=18)
ax1.set_xlabel('year', fontsize=18)
lns2 = ax1.plot(yr1[moving_avg_val/2:-moving_avg_val/2],temp_2,'g',label='Growth-rate model based on:\n'+var_name[np.int(variables_array1b[np.int(input_var1[loc[0]])])]+', '+var_name[np.int(variables_array1b[np.int(input_var2[loc[0]])])]+' and '+var_name[np.int(variables_array1b[np.int(input_var3[loc[0]])])],linewidth=3)
ax2 = ax1.twinx()
lns3 = ax2.plot(yr1[moving_avg_val/2:-moving_avg_val/2],o2,'r', label = 'Shelf-sea model floor O$_2$ (March)',linewidth=3)
plt.ylim(260,282.5)
ax2.set_ylabel('O$_2$ conc. (mmol m$^{-3}$)', fontsize=18)


lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=3,prop={'size':15}).draw_frame(False)


plt.show()

