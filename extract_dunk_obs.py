'''
Script to extract temperature and humidity data from Dunkerswell observations being accumulated on atoll
'''

import glob
import numpy as np

'''
G mph Wind Gust
T C Temperature
V m Visibility
D compass Wind Direction
S mp Wind Speed
W Weather Type
P hpa Pressure
Pt Pa/s Pressure Tendency
Dp C Dew Point
H % Screen Relative Humidity
'''

files = glob.glob('/home/ph290/data0/dunkerswell_obs/*')

year_acc = []
month_acc = []
day_acc = []
hour_acc = []

temperature_acc = []
humidity_acc = []

for file in files:

    f = open(file,'r')
    data = f.read()
    f.close()

    days = data.split('value')
    for i,day in enumerate(days):
        if not 'Wind Gust' in day:
            hours = day.split('</Rep><Rep')
            for hour in hours:
                if '"><Rep' in hour:
                    hour.split('"')
                    time = hour.split('"')[1]
                    tmp = time.split('-')
                    year_tmp = int(tmp[0])
                    month_tmp = int(tmp[1])
                    dat_tmp = int(tmp[2].split('Z')[0])
                year_acc = np.append(year_acc,year_tmp)
                month_acc = np.append(month_acc,month_tmp)
                day_acc = np.append(day_acc,month_tmp)
                hour_tmp =hour.split('Dp="')[1].split('>')[1].split('<')[0]
                hour_acc = np.append(hour_acc,int(hour_tmp)/60.0)
                temperature_tmp = hour.split('T="')[1][0:2]
                humidity_tmp = hour.split('H="')[1][0:4]
                temperature_acc = np.append(temperature_acc,temperature_tmp)
                humidity_acc = np.append(humidity_acc,humidity_tmp)

output_array = np.empty([year_acc.size,6])
output_array[:,0] = year_acc
output_array[:,1] = month_acc
output_array[:,2] = day_acc
output_array[:,3] = hour_acc
output_array[:,4] = temperature_acc
output_array[:,5] = humidity_acc

f = open('/home/ph290/data0/dunkerswell_t_h.txt','w')
f.write('year,month,day,hour,temperature (C),relative humidity (%)\n')
np.savetxt(f,output_array,delimiter=',')
f.close()

#plt.plot(temperature_acc)
#plt.show()

plt.scatter(output_array[:,3],humidity_acc)
plt.show()



