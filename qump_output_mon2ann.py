import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal as signal
from scipy.signal import kaiserord, lfilter, firwin, freqz

'''
Reads monthly qump data in, and converts to annaul averages for annual average box model - adapt soon to do high/low-pass filtering...
'''

in_dir='/project/obgc/qump_out_python/'
filenames=glob.glob(in_dir+'qump_data*.txt')

for files in filenames:
    input=np.genfromtxt(files, delimiter=",")
    test=input.shape
    if test[1] == 2:
        temp_data=np.empty([2099-1860,2])
        temp_data.fill(np.nan)
        for count,i in enumerate(range(1860,2099)):
            index=np.where(np.floor(input[:,0]) == i)
            temp_data[count,:]=[i,np.mean(input[index,1])]
        filename_split = files.split('/')
        np.savetxt(in_dir+'annual_means/'+filename_split[4],temp_data, delimiter=',')
    if test[1] == 5:
        temp_data=np.empty([2099-1860,5])
        temp_data.fill(np.nan)
        for count,i in enumerate(range(1860,2099)):
            index=np.where(np.floor(input[:,0]) == i)
            temp_data[count,:]=[i,np.mean(input[index,1]),np.mean(input[index,2]),np.mean(input[index,3]),np.mean(input[index,4])]
        filename_split = files.split('/')
        np.savetxt(in_dir+'annual_means/'+filename_split[4],temp_data, delimiter=',')

