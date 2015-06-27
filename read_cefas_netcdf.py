'''
NOTE! launch python2.7 with 'sudo' first
'''

import iris
import iris.analysis.cartography
import iris.analysis
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.coords
import iris.quickplot as qplt
import cartopy.crs as ccrs

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


variable_index=1

cefas_dir='/media/Win7_Data/cefas_data/'
cefas_files=['Cefas_hindcast_monthlyvals_site_a.nc','Cefas_hindcast_monthlyvals_site_b.nc','Cefas_hindcast_monthlyvals_site_c.nc','Cefas_hindcast_monthlyvals_site_d.nc']

moll_dir='/media/Win7_Data/butler_data/'
moll_files=['Chickens_IOM_Glycymeris_chronology_1941.txt','Ramsey_IOM_Glycymeris_chronology_1921.txt','IOM_Arctica_chronology_1516.txt','Tiree_Passage_Scotland_Glycymeris_Chronology_1805.txt','north_sea_arctica_f1_chron.txt']
moll_site_names=['Isle of Man 1','Isle of Man 2','Isle of Man 3','Tiree Passage (nr Oban)','N. Sea (N)']

# Isle of Man Arctica from several positions off the west coast of the IOM, but centred around 4 50 E  54 10 N.   Between 25 and 80 metres depth.  For the full range see the map in Butler et al 2009 (EPSL).
# Isle of Man Glycymeris: Chickens is 54 06.031N, 04 23.195W  about 60 metres depth                                   
#                                         Ramsey is 54  06.031N, 04 23.195W about 35-40 metres depth  (These are from the attached paper by Brocas et al)
# North Sea F1 Arctica      59 23.1N, 0 31.0E  about 140m depth (from Butler et al 2009 (Palaeoceanography))
# Tiree Passage Glycymeris  56 37N, 6 24W, 50m water depth   (unpublished)


cube0=iris.load_raw(cefas_dir+cefas_files[0])
cube1=iris.load_raw(cefas_dir+cefas_files[1])
cube2=iris.load_raw(cefas_dir+cefas_files[2])
cube3=iris.load_raw(cefas_dir+cefas_files[3])
cubes=[cube0,cube1,cube2,cube3]

cube=cube0

var_name=[]
for i in np.arange(np.size(cube)):
    var_name.append(cube[i].metadata[1])

coord = cube[0].coord('time')
dt = coord.units.num2date(coord.points)
cfas_year = np.array([coord.units.num2date(value).year for value in coord.points])
cfas_month = np.array([coord.units.num2date(value).month for value in coord.points])
cfas_time=cfas_year+cfas_month/12.0




data=np.genfromtxt(moll_dir+moll_files[4])
mollusk_year=data[:,0]
mollusk_gr=data[:,1]

month=3

tmp=cefas_data=cubes[0].extract(var_name[0])[0].data
monthly_loc=np.where(cfas_month == 3)
monthly_cefas_data=cefas_data[monthly_loc]
temp_data=np.zeros([monthly_cefas_data.size,17])

fig = plt.figure()
for i in np.arange(np.size(cube)):
    ax1 = fig.add_subplot(4,5,i+1)
    for j in np.arange(np.size(cefas_files)):
        cefas_data=cubes[j].extract(var_name[i])[0].data
        monthly_loc=np.where(cfas_month == month)
        monthly_cefas_data=cefas_data[monthly_loc]
        temp_data[:,j]=np.array(monthly_cefas_data)
        unique_years=np.unique(cfas_year)
        ax1.plot(unique_years,movingaverage(monthly_cefas_data,1))

    #ax2 = ax1.twinx()
    #ax2.plot(mollusk_year,movingaverage(mollusk_gr,10), 'r')
    #plt.xlim(1950,2010)
    #plt.title(var_name[i])

plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.2)
plt.show()
print 'month = '+str(month)

'''
'''

fig = plt.figure()
i=5
for j in np.arange(np.size(cefas_files)):
    ax1 = fig.add_subplot(4,1,j)
    #cefas_data=cubes[j].extract(var_name[i])[0].data
    cefas_data=cubes[j].extract('Sea bed oxygen')[0].data
    monthly_loc=np.where(cfas_month == month)
    monthly_cefas_data=cefas_data[monthly_loc]
    unique_years=np.unique(cfas_year)
    ax1.plot(unique_years,movingaverage(monthly_cefas_data,1))

ax2 = ax1.twinx()
ax2.plot(mollusk_year,movingaverage(mollusk_gr,10), 'r')
plt.xlim(1950,2010)
plt.title(var_name[i])

plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.2)
plt.show()
print 'month = '+str(month)

'''
'''

i=16
month=[3]
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#for j in np.arange(np.size(cefas_files)):
    # cefas_data=cubes[j].extract(var_name[i])[0].data
    # monthly_loc=np.where(cfas_month == month)
    # monthly_cefas_data=cefas_data[monthly_loc]
    # unique_years=np.unique(cfas_year)
    # ax1.plot(unique_years,movingaverage(monthly_cefas_data,1), 'b')

cefas_data=cubes[3].extract(var_name[i])[0].data
monthly_loc1=np.where(cfas_month == month[0])
#monthly_loc2=np.where(cfas_month == month[1])
#monthly_loc3=np.where(cfas_month == month[2])
monthly_cefas_data1=cefas_data[monthly_loc1]
#monthly_cefas_data2=cefas_data[monthly_loc2]
#monthly_cefas_data3=cefas_data[monthly_loc3]
monthly_cefas_data=np.mean([monthly_cefas_data1],axis=0)
unique_years=np.unique(cfas_year)

ax1.plot(unique_years,monthly_cefas_data, 'b')
ax1.plot([1987,1987],[-1000,1000], 'g')
#plt.ylim(260,282.5)
plt.ylim(245,315)

ax2 = ax1.twinx()
mv_avg=12
mov_avg_data=movingaverage(mollusk_gr,mv_avg)
ax2.plot(mollusk_year[mv_avg/2.0:-1*(mv_avg/2.0)],mov_avg_data[mv_avg/2.0:-1*(mv_avg/2.0)], 'r')
ax2.plot(mollusk_year,mollusk_gr, 'grey')
plt.ylim(0.0,2.25)
plt.xlim(1860,2010)
ax1.set_ylabel('BFM sea bed oxygen')
ax1.yaxis.label.set_color('blue')
ax2.set_ylabel(' arctica growth rate (10 year moving box smoothed)')
ax2.yaxis.label.set_color('red')
ax1.set_xlabel('year')
plt.show()


'''
'''

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
for j in np.arange(np.size(cefas_files)):
    cefas_data=cubes[j].extract(var_name[i])[0].data
    monthly_loc=np.where(cfas_month == month)
    monthly_cefas_data=cefas_data[monthly_loc]
    unique_years=np.unique(cfas_year)
    ax1.plot(unique_years,movingaverage(monthly_cefas_data,1), 'b')

plt.show()

mollusk_year=[]
mollusk_gr=[]
for i in np.arange(np.size(moll_files)):
    data=np.genfromtxt(moll_dir+moll_files[i])
    mollusk_year.append(data[:,0])
    mollusk_gr.append(data[:,1])



fig=plt.figure()
for i in np.arange(np.size(moll_files)):
    fig.add_subplot(2,3,i+1)
    plt.plot(mollusk_year[i][:],mollusk_gr[i][:])
    plt.title(moll_site_names[i])

plt.show()

fig=plt.figure()
i=3
fig.add_subplot(1,1,1)
moving_avg_val=6
data_tmp=movingaverage(mollusk_gr[i][:],moving_avg_val)
data_tmp=data_tmp[moving_avg_val/2.0,-1*(moving_avg_val/2.0)]
yrs_tmp=mollusk_year[i][:]
yrs_tmp=yrs_tmp[moving_avg_val/2.0,-1*(moving_avg_val/2.0)]
plt.plot(yrs_tmp,data_tmp)
plt.title(moll_site_names[i])

plt.show()


'''
MLR
'''

month=3

tmp=cefas_data=cubes[0].extract(var_name[0])[0].data
monthly_loc=np.where(cfas_month == 3)
monthly_cefas_data=cefas_data[monthly_loc]
temp_data=np.zeros([monthly_cefas_data.size,17])

i=3
for j in np.arange(np.size(cube)):
    cefas_data=cubes[i].extract(var_name[j])[0].data
    monthly_loc=np.where(cfas_month == month)
    monthly_cefas_data=cefas_data[monthly_loc]
    temp_data[:,j]=np.array(monthly_cefas_data)
    unique_years=np.unique(cfas_year)


yr2=mollusk_year[91:]
y=np.array(mollusk_gr[91:])
moving_avg_val=10
y2=movingaverage(np.array(mollusk_gr[91:]),moving_avg_val)
y=y2[moving_avg_val/2:-moving_avg_val/2]

yr1=unique_years[:-5]
x=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,0:16])
o2=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,16])

tmp=np.array([7,12])
x2=np.array(temp_data[moving_avg_val/2:-5-moving_avg_val/2,tmp])

be = np.linalg.lstsq(x,y)
be2 = np.linalg.lstsq(x2,y)

# model is y = b0.x0 + b1.x1 + b2.x2

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(yr1[moving_avg_val/2:-moving_avg_val/2],y)
b=be[0]
x=x
#ax1.plot(yr1[moving_avg_val/2:-moving_avg_val/2],b[0]*x[:,0]+b[1]*x[:,1]+b[2]*x[:,2]+b[3]*x[:,3]+b[4]*x[:,4]+b[5]*x[:,5]+b[6]*x[:,6]+b[7]*x[:,7]+b[8]*x[:,8]+b[9]*x[:,9]+b[10]*x[:,10]+b[11]*x[:,11]+b[12]*x[:,12]+b[13]*x[:,13]+b[14]*x[:,14]+b[15]*x[:,15])
b=be2[0]
x=x2
ax1.plot(yr1[moving_avg_val/2:-moving_avg_val/2],b[0]*x[:,0]+b[1]*x[:,1])

ax2 = ax1.twinx()
ax2.plot(yr1[moving_avg_val/2:-moving_avg_val/2],o2,'r')
plt.ylim(260,282.5)

plt.show()

#can get a lot of teh variability from just 7 and 12
print var_name[7] +' and '+ var_name[12]
