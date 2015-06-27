import iris
import numpy as np
import iris.analysis
import glob

dir = 'ph290@atoll:/home/data/cmip5/canesm2/'

files = glob.glob(dir+'so_*.nc')
print files
