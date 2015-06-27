import glob
import numpy

directory = '/data/data0/ph290/cmip5_data/talk/'

file_names = glob.glob(directory+'*.nc')

models = []
for tmp in file_names:
	models.append(tmp.split('_')[3])

print numpy.unique(models)

#['BNU-ESM' 'CESM1-BGC' 'CMCC-CESM' 'CNRM-CM5' 'CNRM-CM5-2' 'GFDL-ESM2G'
# 'GFDL-ESM2M' 'HadGEM2-CC' 'HadGEM2-ES' 'IPSL-CM5A-LR' 'IPSL-CM5A-MR'
# 'IPSL-CM5B-LR' 'MIROC-ESM' 'MIROC-ESM-CHEM' 'MPI-ESM-LR' 'MPI-ESM-MR'
# 'NorESM1-ME']

