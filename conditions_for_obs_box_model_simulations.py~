import numpy as np
import matplotlib.pyplot as plt


dir_name='/Users/ph290/Documents/MATLAB/co2_files/new/'
in_file='/Users/ph290/Documents/MATLAB/co2_files/qump_data_run_akina_stash_200.txt'
input=np.genfromtxt(in_file, delimiter=",")
output=np.copy(input)

dir_name2='/Users/ph290/Documents/MATLAB/alk_files/new/'
in_file2='/Users/ph290/Documents/MATLAB/alk_files/qump_data_run_akiqj_stash_104.txt'
input2=np.genfromtxt(in_file2, delimiter=",")
output2=np.copy(input2)

tmp_str=np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'])


'''
co2
'''

output[:,1:]=280.0
output2[:,1:]=2300.0

for i in np.arange(4):
	output[:,1:]=280.0
	for j in np.arange(np.size(output[1:,0]))+1:
		if i == 0:
			output[j,1:]=output[j-1,1]
		if i > 0:
			output[j,1:]=output[j-1,1]+output[j-1,1]/(i*100)
	np.savetxt(dir_name+'co2_'+tmp_str[i]+'.txt',output, delimiter=',')
	np.savetxt(dir_name2+'alk_'+tmp_str[i]+'.txt',output2, delimiter=',')
	#plt.plot(output[:,1])
	#plt.show()

'''
alkalinity
'''

output[:,1:]=280.0
output2[:,1:]=2300.0

for i in np.arange(4)+4:
	output2[:,1:]=2300.0
	for j in np.arange(np.size(output2[1:,0]))+1:
		output2[j,1]=output2[j-1,1]-0.1*(i-4)
	np.savetxt(dir_name+'co2_'+tmp_str[i]+'.txt',output, delimiter=',')
	np.savetxt(dir_name2+'alk_'+tmp_str[i]+'.txt',output2, delimiter=',')
	plt.plot(output2[:,1])
	
plt.show()
'''
CO2 and alkalinity
'''

output[:,1:]=280.0
output2[:,1:]=2300.0

for i in np.arange(4)+8:
	output[:,1:]=280.0
	output2[:,1:]=2300.0
	for j in np.arange(np.size(output[1:,0]))+1:
		if i == 8:
			output[j,1:]=output[j-1,1]
		output[j,1:]=output[j-1,1]+output[j-1,1]/(2*100)
		if i > 8:
			output2[j,1]=output2[j-1,1]-0.1*(i-8)
	np.savetxt(dir_name+'co2_'+tmp_str[i]+'.txt',output, delimiter=',')
	np.savetxt(dir_name2+'alk_'+tmp_str[i]+'.txt',output2, delimiter=',')
	#plt.plot(output[:,1])
	#plt.show()

	
