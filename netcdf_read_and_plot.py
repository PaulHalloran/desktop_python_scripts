#note that this does not yet work for non regular grids

from mpl_toolkits.basemap import Basemap, cm
from netCDF4 import Dataset as NetCDFFile
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import numpy.ma as ma

def read_and_plot(file_name):
#function to read ans plot CMIP5 data

    print file_name.strip()
    nc = NetCDFFile(file_name.strip())
    
        # data from http://water.weather.gov/precip/
    var = nc.variables['fgco2']
    var1=var[0]
    data = np.fliplr(var1[:])
    lats = nc.variables['lat'][:]
    lons = -nc.variables['lon'][:]
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plt.contourf(lons, lats, data, 60,
                 transform=ccrs.PlateCarree())
    
    ax.coastlines()
    plt.title(model[temp[0]])
    outfile = '/net/home/h04/hador/temp_plots/'+model[temp[0]].strip()+'_test.png'
    plt.savefig(outfile)
    plt.show()

'''
main code - find the first air-sea CO2 flux file from each CMIP5 model
'''

#Read in the names of all of the CMIP5 files with the variable 'fgco2'
file_data=[]
p = Popen("/project/champ/mec/bin/managecmip list-local -e rcp85 -f mon -v fgco2",stdout=PIPE,stderr=PIPE,shell=True)

for line in p.stdout:
    file_data.append(line.split('/'))
for err in p.stderr:
    pass

#split the long file-names into sections to pull out model names, variable names etc.
model_centers = []
model= []
run_name= []
variable= [] 

for temp in file_data:
    model_centers.append(temp[6])
    model.append(temp[7])
    run_name.append(temp[8])
    variable.append(temp[14])  

#come up with a unique list of models
unique_model=np.unique(np.array(model))

#lind the first items in the list of all the model data files relating to each separate model
model_loc_base=[]
for temp_model in unique_model:
    model_loc=[]
    for i,x in enumerate(model):
        if x == temp_model:
            model_loc.append(i)
    model_loc_base.append(model_loc)
    
#read in and plot the model data
for i,temp_model in enumerate(unique_model):
    temp=model_loc_base[i]     
    file_name = '/'.join(file_data[temp[0]])
    #test = file_name.find('CESM1-BGC')
    #if test == -1:
    read_and_plot(file_name)

