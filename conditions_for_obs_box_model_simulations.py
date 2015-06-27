import numpy as np
import matplotlib.pyplot as plt

'''
reading in data
'''

t_file='/home/ph290/data1/qump_out_python/annual_means/qump_data_run_akina_stash_101.txt'
input1=np.genfromtxt(t_file, delimiter=",")
output1=np.copy(input1)
output1[:,1]=6.0 #SPG
output1[:,2]=20.0 #STG
output1[:,3]=10.0 #South
output1[:,4]=10.0

s_file='/home/ph290/data1/qump_out_python/annual_means/qump_data_run_akina_stash_102.txt'
input2=np.genfromtxt(s_file, delimiter=",")
output2=np.copy(input2)
output2[:,1]=-8.0e-04
output2[:,2]=-5.0e-04
output2[:,3]=8.0e-04
output2[:,4]=1.0e-04

a_file='/home/ph290/data1/qump_out_python/annual_means/qump_data_run_akina_stash_104.txt'
input3=np.genfromtxt(a_file, delimiter=",")
output3=np.copy(input3)
output3[:,1]=2350.0
output3[:,2]=2400.0
output3[:,3]=2450.0
output3[:,4]=2400.0

c_file='/home/ph290/data1/qump_out_python/annual_means/qump_data_run_akina_stash_200.txt'
input4=np.genfromtxt(c_file, delimiter=",")
output4=np.copy(input4)

m_file='/home/ph290/data1/qump_out_python/annual_means/qump_data_run_akina_moc_stm_fun.txt'
input5=np.genfromtxt(m_file, delimiter=",")
output5=np.copy(input5)
output5[:,1]=18.0

rcp85_file='/home/ph290/data0/misc_data/RCP85_MIDYR_CONC.DAT'
input_co2=np.genfromtxt(rcp85_file,skip_header=40)
#plt.plot(input_co2[:,0],input_co2[:,3])

#getting rcp8.5 data to use for CO2 scenario
for i in np.arange(np.size(input4[:,0])):
	loc=np.where(input_co2[:,3] == input4[i,0])
	if np.size(loc) == 1:
		output4[i,1:]=input_co2[loc,3]

'''
writing data out
'''

np.savetxt('/home/ph290/data0/matlab_data/s_101.txt',output1, delimiter=',')
np.savetxt('/home/ph290/data0/matlab_data/s_102.txt',output2, delimiter=',')
np.savetxt('/home/ph290/data0/matlab_data/s_104.txt',output3, delimiter=',')
np.savetxt('/home/ph290/data0/matlab_data/s_200.txt',output4, delimiter=',')
np.savetxt('/home/ph290/data0/matlab_data/s_moc_stm_fun.txt',output5, delimiter=',')


