import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib


array_size=239

source=r"/net/project/obgc/boxmodel_testing_less_boxes_4_monthly_inclusing_vol_params/results/order_runs_processed_in_box_model.txt"
inp = open(source,"r")
count=0
for line in inp:
    count +=1

'''
Read in subpolar gyre co2 flux data from QUMP
'''

run_names_order=str.split(line,' ')
run_names_order=run_names_order[0:-1]

input_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)


dir0='/net/project/obgc/qump_out_python/annual_means/'
no_filenames=glob.glob(dir0+'*30249.txt')
filenames0=glob.glob(dir0+'*'+run_names_order[1]+'*30249.txt')

qump_year=np.zeros(array_size)
qump_co2_flux=np.zeros(array_size)

input2=np.genfromtxt(filenames0[0], delimiter=",")
qump_year=input2[:,0]

filenames0=glob.glob(dir0+'*'+run_names_order[1]+'*30249.txt')
input2=np.genfromtxt(filenames0[0], delimiter=",")
if input2[:,1].size == array_size:
     qump_co2_flux[:]=input2[:,1]

'''
Read in data from box model
'''

dir1='/net/project/obgc/boxmodel_testing_less_boxes_4_annual_b/results/'
filenames=glob.glob(dir1+'box_model_qump_results_20.csv')
input1=np.genfromtxt(filenames[0], delimiter=",")
box_co2_flux=np.zeros(array_size)
box_years=input1[:,0]
box_co2_flux[:]=input1[:,2]

'''
'''

dir2='/net/project/obgc/boxmodel_testing_less_boxes_4_annual_b/results2/'
filenames2=glob.glob(dir2+'box_model_qump_results_20.csv')
input2=np.genfromtxt(filenames2[0], delimiter=",")
box_co2_flux2=np.zeros(array_size)
box_years2=input2[:,0]
box_co2_flux2[:]=input2[:,2]

'''
plotting
'''

smoothing_len=12.0
plt.plot(box_years[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg(qump_co2_flux,smoothing_len),'k')
plt.plot(box_years[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg((box_co2_flux/1.12e13)*1.0e15/12.0,smoothing_len),'b')
plt.plot(box_years[smoothing_len/2.0:-smoothing_len/2.0+1],matplotlib.mlab.movavg((box_co2_flux2/1.12e13)*1.0e15/12.0,smoothing_len),'r')
plt.show()
