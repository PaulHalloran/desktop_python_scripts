import numpy as np
import numpy.ma as ma
import pylab 
import scipy.stats as stats
import carbchem
import matplotlib.pyplot as plt
import iris
import os
import glob

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


# obs_file = '/data/local/hador/obs/CARINA/CARINA.ATL.V1.0.csv'

# obs_data = np.genfromtxt(obs_file,skip_header=1,usecols=[3,4,5,6,12,13,14,22,23], delimiter=",")
# mdi=-999.0

# month_col=0
# year_col=1
# latitude_col=2
# longitude_col=3
# depth_col=4
# temperature_col=5
# salinity_col=6
# tco2_col=7
# alk_col=8

# years=obs_data[:,year_col]
# months=obs_data[:,month_col]
# t=np.copy(obs_data[:,temperature_col])
# s=np.copy(obs_data[:,salinity_col])
# tco2=np.copy(obs_data[:,tco2_col])
# tco2[tco2 != mdi] *= 1.0e-6
# talk=np.copy(obs_data[:,alk_col])
# talk[talk != mdi] *= 1.0e-6
# depth=np.copy(obs_data[:,depth_col])

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
#     for month_temp in np.arange(1,13):
#         #yr_temp=1977.0 #testing
#         #month_temp=10
#         #identifies where that year and month contans all the data required to calculate pCO2 AND has a depth less than or equal to 20m
#         lats_temp=lats[(years == yr_temp) & (months == month_temp) & (tco2 != mdi) & (t != mdi) & (s != mdi) & (talk != mdi) & (depth <= 20.0)]
#         lons_temp=lons[(years == yr_temp) & (months == month_temp) & (tco2 != mdi) & (t != mdi) & (s != mdi) & (talk != mdi) & (depth <= 20.0)]
#         t_temp=t[(years == yr_temp) & (months == month_temp) & (tco2 != mdi) & (t != mdi) & (s != mdi) & (talk != mdi) & (depth <= 20.0)]
#         s_temp=s[(years == yr_temp) & (months == month_temp) & (tco2 != mdi) & (t != mdi) & (s != mdi) & (talk != mdi) & (depth <= 20.0)]
#         tco2_temp=tco2[(years == yr_temp) & (months == month_temp) & (tco2 != mdi) & (t != mdi) & (s != mdi) & (talk != mdi) & (depth <= 20.0)]
#         talk_temp=talk[(years == yr_temp) & (months == month_temp) & (tco2 != mdi) & (t != mdi) & (s != mdi) & (talk != mdi) & (depth <= 20.0)]
#         #pulls out only those lats and longs corresponding to the subpolar gyre
#         test1=np.in1d(lats_temp,spg_lats)
#         test2=np.in1d(lons_temp,spg_lons)
 
#         #note - I can't find a simple way of combining the two masks - this is the best I can manage!
#         combined_mask=np.copy(test1)
#         combined_mask.fill(False)
#         loc=[i for i,x in enumerate(test1) if (test1[i] == True) & (test2[i] == True)]
#         combined_mask[loc]=True
  
#         #calculate the mean pco2 for the spg in that year and month
#         if t_temp[combined_mask].size > 1:
#             temp_co2_data = carbchem.carbchem(1,mdi,t_temp[combined_mask],s_temp[combined_mask],tco2_temp[combined_mask],talk_temp[combined_mask])
#             # print 'lats: '+str(lats_temp[combined_mask[0:10]])
#             # print 'lons: '+str(lons_temp[combined_mask[0:10]])
#             # print 'T: '+str(t_temp[combined_mask[0:10]])
#             # print 'S: '+str(s_temp[combined_mask[0:10]])
#             # print 'TALK: '+str(tco2_temp[combined_mask[0:10]])
#             # print 'TCO2: '+str(talk_temp[combined_mask[0:10]])
#             # print np.mean(temp_co2_data)
#             # print 'year: '+str(yr_temp)
#             # print 'month: '+str(month_temp)
#             # print '\n'
#             co2_obs.append(np.mean(temp_co2_data))
#             year_co2.append(yr_temp)
#             month_co2.append(month_temp)
#             lats_obs.append(lats_temp[combined_mask])
#             lons_obs.append(lons_temp[combined_mask])
#         if t_temp[combined_mask].size == 1:
#             temp_co2_data = carbchem.carbchem(1,mdi,t_temp[combined_mask].repeat(2),s_temp[combined_mask].repeat(2),tco2_temp[combined_mask].repeat(2),talk_temp[combined_mask].repeat(2))
#             co2_obs.append(temp_co2_data[0])
#             year_co2.append(yr_temp)
#             month_co2.append(month_temp)
#             lats_obs.append(lats_temp[combined_mask])
#             lons_obs.append(lons_temp[combined_mask])

# plt.plot(np.array(year_co2)+np.array(month_co2)/12.0,np.array(co2_obs),'o')
# plt.show()


'''
now read in QUMP data and pull out data from points corresponding to where we have obs.
'''

#problem - iris appears to be unable to read in this data...
#now just meteasplitting into different stashes to hopefully get round this problem - see ~/IDL/metasplit_stuff.pro

dir ='/data/local2/qump_n_atl_mor_var_monthly/'

ensemble_co2=[]
#ensemble_co2 will hold all of the relevant co2 values for all of the enseble members

os.chdir(dir)
#first loop through different model runs
for directory in glob.glob("a*"):
    test=glob.glob(dir+directory+'/*000001000000*')
    if test == []:
        print 'not stashsplit files, skipping to next directory'
        continue
    file_names=[]
    file_year=[]
    file_month=[]
    os.chdir(dir+directory+'/')
    #pull our relevant files second loop through 
    for file in glob.glob("000001000000.30.248*.pp"):
        file_temp = file.split('.')
        file_year.append(int(file_temp[4]))
        file_month.append(int(file_temp[5]))
        file_names.append(file)
    model_co2=[]
    #loop through files of interest
    for i,temp_year in enumerate(year_co2):
        temp_model_co2=[]
        temp_month=month_co2[i]
        temp_year=1879 # testing
        temp_month=12 # testing
        #nice little bit of code to find the common array index for the year and month in arrays
        try:
            year_index = file_year.index(temp_year)
            month_index = file_month.index(temp_month)
            common_index = set([year_index]) & set([month_index])
        except ValueError:
            continue
            #skips to the next iteration of loop if no index is found
        common_index2=list(common_index)
        if len(common_index2) > 0:
            #print file_names[np.array(common_index2)]
            temp_cube = iris.load_cube(dir+directory+'/'+file_names[np.array(common_index2)])
            qump_lats=temp_cube.coord("latitude").points
            qump_lons=temp_cube.coord("longitude").points
            #now need to just find the nearest lat and lon to the obs, then pluck out the data:
            for j,x in enumerate(lats_obs[i]):
                lat_index = find_nearest(qump_lats, lats_obs[i][j])
                lon_index = find_nearest(qump_lons-180.0, lons_obs[i][j])
                temp_model_co2.append(temp_cube.data[lat_index,lon_index])
                #print 'lat: '+str(qump_lats[lat_index])+', lon: '+str(qump_lons[lon_index]-180)+' co2: '+str(temp_cube.data[lat_index,lon_index]) # debugging
            mask=~np.isnan(temp_model_co2)
            model_co2.append(np.mean(np.array(temp_model_co2)[mask]))
    ensemble_co2.append(model_co2)


#then do Q-Q plots. This is well simpl - just sort the data into order ad plot  one against the other. If you want you can turn the sorted data into percentiles to make it easier to understand.

#plt.plot(sort(co2_obs),MODEL DATA WHEN DOWNLOADED)
#plt.show()
