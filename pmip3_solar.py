import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import running_mean as rm
import running_mean_post as rmp
import iris
import iris.quickplot as qplt
import scipy
import scipy.stats
import statsmodels.api as sm


start_date = 950
#start_date = 1400
end_date = 1849
end_date = 1200


N=5.0
#I think N is the order of the filter - i.e. quadratic
timestep_between_values=1.0 #years valkue should be '1.0/12.0'
low_cutoff=100.0
high_cutoff=10.0

Wn_low=timestep_between_values/low_cutoff
Wn_high=timestep_between_values/high_cutoff

b, a = scipy.signal.butter(N, Wn_low, btype='high')
b1, a1 = scipy.signal.butter(N, Wn_high, btype='low')

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)
data4 = np.genfromtxt(file4)

data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]
data = np.mean(data_tmp,axis = 1)
voln_n = data1.copy()
voln_n[:,1] = data

data_tmp[:,0] = data3[:,1]
data_tmp[:,1] = data4[:,1]
data = np.mean(data_tmp,axis = 1)
voln_s = data1.copy()
voln_s[:,1] = data


amo_file = '/home/ph290/data0/misc_data/iamo_mann.dat'
amo = np.genfromtxt(amo_file, skip_header = 4)
amo_yr = amo[:,0]
amo_data = amo[:,1]
loc = np.where((amo_yr <= end_date) & (amo_yr >= start_date))
amo_yr = amo_yr[loc]
amo_data = amo_data[loc]
amo_data = scipy.signal.filtfilt(b, a, amo_data)
x = amo_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
amo_data = x

directory = '/home/ph290/data0/misc_data/last_millenium_solar/'
    
file1 = directory+'tsi_VK.txt'
file3 = directory+'tsi_SBF_11yr.txt' 
file4 = directory+'tsi_DB_lin_40_11yr.txt' 


data1 = np.genfromtxt(file1,skip_header = 4)
data1_yr = data1[:,0]
loc = np.where((data1_yr <= end_date) & (data1_yr >= start_date))
data1_yr = data1[loc[0],0]
data1_data = data1[loc[0],1]
data1_data = scipy.signal.filtfilt(b, a, data1_data)
x = data1_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
data1_data = x


data3 = np.genfromtxt(file3,skip_header = 4)
data3_yr = data3[:,0]
loc = np.where((data3_yr <= end_date) & (data3_yr >= start_date))
data3_yr = data3[loc[0],0]
data3_data = data3[loc[0],1]
data3_data = scipy.signal.filtfilt(b, a, data3_data)
x = data3_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
data3_data = x

data4 = np.genfromtxt(file4,skip_header = 4)
data4_yr = data4[:,0]
loc = np.where((data4_yr <= end_date) & (data4_yr >= start_date))
data4_yr = data4[loc[0],0]
data4_data = data4[loc[0],1]
data4_data = scipy.signal.filtfilt(b, a, data4_data)
x = data4_data
x=(x-np.min(x))/(np.max(x)-np.min(x))
data4_data = x



smth =7
smth2 =7
volc_yr = voln_n[:,0]
loc_v = np.where((volc_yr <= end_date) & (volc_yr >= start_date))
volc_yr_II = volc_yr[loc_v]

tmp = data3_data
tmp_yr = data3_yr
tmp = scipy.signal.filtfilt(b, a, tmp)
tmp = scipy.signal.filtfilt(b1, a1, tmp)

data1_int = np.interp(volc_yr_II,tmp_yr,tmp)
data1_int = rmp.running_mean_post(data1_int,smth2*36.0)
#data1_int[np.where(np.isnan(data1_int))] = 0.0
mann_amo = np.interp(volc_yr_II,amo_yr,amo_data)
vns = rmp.running_mean_post(voln_n[loc_v[0],1],smth*36.0)

x1 = vns
x2 = data1_int
y = mann_amo
x = np.column_stack((x1,x2))
#stack explanatory variables into an array
x = sm.add_constant(x)
#add constant to first column for some reasons
model = sm.OLS(y,x)
results = model.fit()

x1b = vns
yb = mann_amo
#stack explanatory variables into an array
xb = sm.add_constant(x1b)
#add constant to first column for some reasons
modelb = sm.OLS(yb,xb)
resultsb = modelb.fit()


smooth_var=5

plt.close('all')
fig = plt.figure(figsize=(13,6))
ax1 = fig.add_subplot(111)
#ax1.plot(data1_yr,rmp.running_mean_post(data1_data,smooth_var))
#ax1.plot(data3_yr,rmp.running_mean_post(data3_data,smooth_var))
#ax1.plot(data4_yr,rmp.running_mean_post(data4_data,smooth_var))

ax1.plot(amo_yr,amo_data,'k',linewidth = 3,alpha = 0.75)

ax1.plot(volc_yr_II,results.params[2]*x2+results.params[1]*x1+results.params[0],'b',linewidth = 3,alpha = 0.75)
ax1.plot(volc_yr_II,resultsb.params[1]*x1+resultsb.params[0],'r',linewidth = 3,alpha = 0.75)
plt.show(block = False)
