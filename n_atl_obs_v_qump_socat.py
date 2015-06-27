import numpy as np
import numpy.ma as ma
import pylab 
import scipy.stats as stats
import matplotlib.pyplot as plt
import iris
import os
import glob
import math
from pylab import *

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


#obs_file = '/data/local/hador/obs/CARINA/CARINA.ATL.V1.0.csv'
obs_file = '/project/obgc/SOCATv1.5_NorthAtlantic_2011_09_22.txt'

# obs_data = np.genfromtxt(obs_file,skip_header=40,usecols=[2,3,7,8,18], delimiter="\t")
# mdi=-999.0

# month_col=1
# year_col=0
# latitude_col=3
# longitude_col=2
# co2_col=4

# years=obs_data[:,year_col]
# months=obs_data[:,month_col]
# co2=np.copy(obs_data[:,co2_col])

# min_year=np.min(years)
# max_year=np.max(years)

# lats=np.round(obs_data[:,latitude_col])
# lons=np.round(obs_data[:,longitude_col])

# '''
# now bring in data about longhurst province
# '''
# longhurst_file='/home/h04/hador/data/BGCP_Longhurst.csv'
# longhurst_data = np.genfromtxt(longhurst_file,usecols=[0,1,2],delimiter=",")
# longhurst_num=longhurst_data[:,0]
# longhurst_lon=longhurst_data[:,1]
# longhurst_lat=longhurst_data[:,2]

# spg_lats=longhurst_lat[(longhurst_num == 2) | (longhurst_num == 4)]
# spg_lons=longhurst_lon[(longhurst_num == 2) | (longhurst_num == 4)]
# #note longurst procinves 2 and 4 relate to the SPG
# #note | is the way to say 'or' here

# spg_lats_old=np.array(spg_lats)
# spg_lons_old=np.array(spg_lons)

# spg_lats=spg_lats_old[spg_lats_old < 66]
# spg_lons=spg_lons_old[spg_lats_old < 66]
# # #removing the arctic from the spg



# '''
# now identify subpolar gyre locations with T, S, TALK and TCO2 in each month and calculate the pCO2 
# '''

# year_co2=[]
# month_co2=[]
# co2_obs=[]
# lats_obs=[]
# lons_obs=[]

# #count through all of the years and months
# for yr_temp in np.arange(((max_year-min_year)+1))+min_year:
#     print 'reading obs for year: '+str(yr_temp)
#     for month_temp in np.arange(1,13):
#         #yr_temp=1996.0 #testing
#         #month_temp=5
#         #identifies where that year and month and co2 contans data
#         lats_temp=lats[(years == yr_temp) & (months == month_temp) & (co2 != mdi)]
#         lons_temp=lons[(years == yr_temp) & (months == month_temp) & (co2 != mdi)]
#         co2_temp=co2[(years == yr_temp) & (months == month_temp) & (co2 != mdi)]
#         #pulls out only those lats and longs corresponding to the subpolar gyre
#         test1=np.in1d(lats_temp,spg_lats)
#         test2=np.in1d(lons_temp,spg_lons)
#         #note - I can't find a simple way of combining the two masks - this is the best I can manage!
#         combined_mask=np.copy(test1)
#         combined_mask.fill(False)
#         loc=[i for i,x in enumerate(test1) if (test1[i] == True) & (test2[i] == True)]
#         combined_mask[loc]=True
#         #calculate the mean pco2 for the spg in that year and month
#         if co2_temp[combined_mask].size > 0:
#             co2_obs.append(np.mean(co2_temp[combined_mask]))
#             year_co2.append(yr_temp)
#             month_co2.append(month_temp)
#             lats_obs.append(lats_temp[combined_mask])
#             lons_obs.append(lons_temp[combined_mask])

# #plt.plot(np.array(year_co2)+np.array(month_co2)/12.0,np.array(co2_obs))
# #plt.show()



# '''
# Now read in QUMP data and pull out data from points corresponding to where we have obs.
# '''

# #problem - iris appears to be unable to read in this data...
# #now just meteasplitting into different stashes to hopefully get round this problem - see ~/IDL/metasplit_stuff.pro

# dir ='/data/local2/qump_n_atl_mor_var_monthly/'

# ensemble_co2=[]
# #ensemble_co2 will hold all of the relevant co2 values for all of the ensemble members

# os.chdir(dir)
# #first loop through different model runs
# for directory in glob.glob("a*"):
#     #directory='akoxp'
#     print 'processing directory: '+directory
#     test=glob.glob(dir+directory+'/*000001000000*')
#     if test == []:
#         print 'not stashsplit files, skipping to next directory'
#         continue
#     file_names=[]
#     file_year=[]
#     file_month=[]
#     os.chdir(dir+directory+'/')
#     #pull our relevant files second loop through 
#     for file in glob.glob("000001000000.02.30.248*.pp"):
#         file_temp = file.split('.')
#         file_year.append(int(file_temp[5]))
#         file_month.append(int(file_temp[6]))
#         file_names.append(file)
#     model_co2=[]
#     #loop through files of interest
#     #back to here
#     for i,temp_year in enumerate(year_co2):
#         temp_model_co2=[]
#         temp_month=month_co2[i]
#         #temp_year=1999 # testing
#         #temp_month=7 # testing
#         #nice little bit of code to find the common array index for the year and month in arrays
#         try:
#             year_index = [j for j,val in enumerate(file_year) if val==temp_year]
#             month_index = [j for j,val in enumerate(file_month) if val==temp_month]
#             common_index = set(year_index) & set(month_index)
#         except ValueError:
#             continue
#             #skips to the next iteration of loop if no index is found
#         common_index2=list(common_index)
#         if len(common_index2) > 0:
#             #print file_names[np.array(common_index2)]
#             temp_cube = iris.load_cube(dir+directory+'/'+file_names[np.array(common_index2)])
#             qump_lats=temp_cube.coord("latitude").points
#             qump_lons=temp_cube.coord("longitude").points
#             #now need to just find the nearest lat and lon to the obs, then pluck out the data:
#             #back to here
#             for j,x in enumerate(lats_obs[i]):
#                 lat_index = find_nearest(qump_lats, lats_obs[i][j])
#                 lon_index = find_nearest(qump_lons-180.0, lons_obs[i][j])
#                 temp_model_co2.append(temp_cube.data[lat_index,lon_index])
#                   #print 'lat: '+str(qump_lats[lat_index])+', lon: '+str(qump_lons[lon_index]-180)+' co2: '+str(temp_cube.data[lat_index,lon_index]) # debugging
#         mask=~np.isnan(temp_model_co2)
#         model_co2.append(np.mean(np.array(temp_model_co2)[mask]))
#             #if np.mean(np.array(temp_model_co2)[mask]) < 200:
#                 #print lats_obs[i]
#                 #print lons_obs[i]
#                 #print np.array(temp_model_co2)[mask]
#     ensemble_co2.append(model_co2)


#then do Q-Q plots. This is well simple - just sort the data into order ad plot  one against the other. If you want you can turn the sorted data into percentiles to make it easier to understand.

#execfile('/home/h04/hador/python_scripts/n_atl_obs_v_qump_socat.py')


ensemble_co2=np.array(ensemble_co2)

where_values=np.where(np.isfinite(ensemble_co2[:,0]))
masking=np.ma.masked_invalid(np.array(ensemble_co2[where_values[0][0]]))
mask_indicies=np.where(masking.mask == False)

array_shape=ensemble_co2.shape
ensemble_co2_b=np.zeros([np.size(where_values),np.size(mask_indicies)])
for i,x in enumerate(where_values[0]):
    ensemble_co2_b[i]=ensemble_co2[x,mask_indicies[0]]


ensemble_co2_sorted=np.array(ensemble_co2_b)
for i,x in enumerate(range((ensemble_co2_b[:,0]).size)):
    ensemble_co2_sorted[i]=np.sort(ensemble_co2_b[i])

co2_obs_b=np.array(co2_obs)
co2_obs_b=co2_obs_b[mask_indicies[0]]
co2_obs_sorted=np.sort(co2_obs_b)

fig=plt.figure()
ax=fig.add_subplot(212)
for i,x in enumerate(range(np.size(ensemble_co2_b[:,0]))):
    test=np.size(ensemble_co2_sorted[i])
    if test==np.size(co2_obs_sorted):
        y=ensemble_co2_sorted[i,np.isfinite(ensemble_co2_sorted[i,:])]
        x=co2_obs_sorted[np.isfinite(ensemble_co2_sorted[i,:])]
        ax.plot(x,y,'k*',markerfacecolor='white',markeredgewidth=0.5)
        #ax.scatter(co2_obs_sorted,ensemble_co2_sorted[i],s=10, facecolors='none', edgecolors='k')


finite_vars=[]
for i,x in enumerate(range(np.size(ensemble_co2_sorted[:,0]))):
    masked = np.ma.masked_array(ensemble_co2_sorted[i],np.isnan(ensemble_co2_sorted[i]))
    temp = np.mean(masked)
    if temp > 0.0:
        finite_vars.append(i)

i=finite_vars[1]
ax.plot(co2_obs_sorted,ensemble_co2_sorted[i],'k*',markerfacecolor='red',markeredgewidth=0.5)
i=finite_vars[1]+9
ax.plot(co2_obs_sorted,ensemble_co2_sorted[i],'k*',markerfacecolor='green',markeredgewidth=0.5)
i=finite_vars[1]+19
ax.plot(co2_obs_sorted,ensemble_co2_sorted[i],'k*',markerfacecolor='blue',markeredgewidth=0.5)

ax.plot(np.array([0,1000]),np.array([0,1000]),'k')
ax.set_xlim(200,550)
ax.set_ylim(200,550)
ax.set_xlabel('sorted pCO$_2$ observations')
ax.set_ylabel('sorted model pCO$_2$')
ax.set_title('Q-Q plot')

dates=np.array(year_co2)+np.array(month_co2)/12.0
dates_b=dates[mask_indicies[0]]

bx=fig.add_subplot(211)
bx.scatter(dates_b,co2_obs_b,facecolors='k',s=20, edgecolors='k',linewidth=0.5,label='observations')
#y=np.array(co2_obs)
#x=np.array(year_co2)+np.array(month_co2)/12.0
#m,c=polyfit(x,y,1)
#bx.plot(x,m*x+c, '--k')

i=finite_vars[1]+0
bx.scatter(dates_b,ensemble_co2_b[i],s=20, facecolors='none', edgecolors='r',linewidth=0.5,label='e.g. model ensemble member 1')
#y=ma.masked_invalid(np.array(ensemble_co2[i]))
#x=np.array(year_co2)+np.array(month_co2)/12.0
#mask=np.logical_not(y.mask)
#x=x[mask]
#y=y[mask]
#m,c=polyfit(x,y,1)
#bx.plot(x,m*x+c, '--r')

i=finite_vars[1]+11
bx.scatter(dates_b,ensemble_co2_b[i],s=20, facecolors='none', edgecolors='g',linewidth=0.5,label='e.g. model ensemble member 11')

i=finite_vars[1]+21
bx.scatter(dates_b,ensemble_co2_b[i],s=20, facecolors='none', edgecolors='b',linewidth=0.5,label='e.g. model ensemble member 21')

bx.legend(loc=2,prop={'size':10})
leg = plt.gca().get_legend()
leg.draw_frame(False)
bx.set_xlabel('Year')
bx.set_ylabel('pCO$_2$')
bx.set_ylim(200,600)
bx.set_xlim(1980,2006)
bx.set_title('Model subsampled at times and locations of observations')

subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,wspace=0.0, hspace=0.4)

plt.savefig('/home/h04/hador/public_html/twiki_figures/qump_v_obs_spg_pco2_variability.ps')
plt.savefig('/home/h04/hador/public_html/twiki_figures/qump_v_obs_spg_pco2_variability.png')
plt.show()

print 'Model subsampled at lat, lon, month and year of each observation in socat, relating to the two longhurst provinces defining the subpolar gyre, and limited to below 65 degrees North. Obs and MOdel results are then averaged to a single value for each month. Three arbitrary model ensemble members are displayed in the top figure for visual comparison, then a Q-Q pliot is produced with all of the data in the second planel. For this the data presented above (but for all enemble members) is simply sorted into assending order, then plotted againts the obs, which are also sorted into ascending order of CO2 partial pressure.'
