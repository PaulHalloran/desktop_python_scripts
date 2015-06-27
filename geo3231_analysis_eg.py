from iris import *
from iris.analysis import *
import iris.analysis
import iris.quickplot as qplt
from numpy import *
from matplotlib.pyplot import *
from scipy.stats.mstats import *
import iris.plot as iplt
import seawater


cfc_file = '/home/ph290/Documents/teaching/glodap/pCFC11/pCFC11.nc'
# the file I want to read in

cfc_cube = load_cube(cfc_file,'pCFC-11')
#reading in the file

'''
We'll focus on the Atlantic to start with
'''

atlantic_region = Constraint(longitude=lambda v: -70 <= v <= 0,latitude=lambda v: -90 <= v <= 90)
#make a variable that holds information about the data points I should use if I want to look at the Atlantic region (here defined as between 60 and 0W and 50S and 70N)

cfc_atlantic = cfc_cube.extract(atlantic_region)
#extract the data relating to each region

'''
And in the atlantic we'll start by looking at the surface ocean
'''

cfc_atlantic_surface = cfc_atlantic.extract(Constraint(depth = 0))
#extract the surface level data

figure()
qplt.contourf(cfc_atlantic_surface,linspace(0,300,50))
gca().coastlines()
savefig('/home/ph290/Documents/figures/atl_surface_cfc.png')
show()
#this looks good, we we've saved it

'''
Next we'll produce a plot with depth gainst latitude, averaging all o fthe Atlantic longitude points together
'''

cfc_atlantic_meridional = cfc_atlantic.collapsed('longitude',MEAN)
#this averages all of teh longitdes together. Next we can plot this

figure()
qplt.contourf(cfc_atlantic_meridional,linspace(0,300,50))
savefig('/home/ph290/Documents/figures/atl_meridional_cfc.png')
show()
#this looks good, we we've saved it

'''
So can we explain why there values are low at the surface in teh Southern Ocean, and high in the subsurface in the high-latitude N. Atlantic?

Let's start by looking at the temperature and salinity to see if by thiking about the density of the water we can understand what is going on
'''

temperature_file = '/home/ph290/Documents/teaching/temperature_annual_1deg.nc'
salinity_file = '/home/ph290/Documents/teaching/salinity_annual_1deg.nc'

temperature_cube = load_cube(temperature_file,'Statistical Mean')[0]
salinity_cube = load_cube(salinity_file,'Statistical Mean')[0]

#note here I'm just changing the name in the metadata, because 'statistical mean' is not very helpful
temperature_cube.long_name='temperature'
salinity_cube.long_name='salinity'

atlantic_region2 = Constraint(longitude=lambda v: 360-70 <= v <= 360,latitude=lambda v: -90 <= v <= 90)
#note that the temperature and salinity data have different latitude values than the cfc data - now they go from 0 to 360 rather than -360 to 0. Annoying, but that is just how it is... You can check this by typing  
# print temperature_cube.coord('longitude').points
#the consequence is that we have to redifine teh Atlantic regoin, and use that region instead

temperature_atlantic = temperature_cube.extract(atlantic_region2)
temperature_atlantic_meridional = temperature_atlantic.collapsed('longitude',MEAN)
figure()
qplt.contourf(temperature_atlantic_meridional,linspace(-4,30,50))
savefig('/home/ph290/Documents/figures/atl_meridional_temperature.png')
show()

salinity_atlantic = salinity_cube.extract(atlantic_region2)
salinity_atlantic_meridional = salinity_atlantic.collapsed('longitude',MEAN)
figure()
qplt.contourf(salinity_atlantic_meridional,linspace(33,38,50))
savefig('/home/ph290/Documents/figures/atl_meridional_salinity.png')
show()

#We can use the above, and the python package 'seawater' to calculate seawaters density for us 

density_cube = temperature_cube.copy()
#first we need to make a new cube to hold the density data when we calculate it. This is what we do here. 

#we also want to give that cube the right metadata
density_cube.standard_name = 'sea_water_density'
density_cube.units = 'kg m-3'

density_cube.data = seawater.dens(salinity_cube.data,temperature_cube.data,1)
density_atlantic = density_cube.extract(atlantic_region2)
density_atlantic_meridional = density_atlantic.collapsed('longitude',MEAN)

figure()
qplt.contourf(cfc_atlantic_meridional,linspace(0,300,50))
CS=iplt.contour(density_atlantic_meridional,array([1026.2,1026.4,1027.6,1026.8,1027.0,1027.2,1027.3,1027.4,1027.5,1027.6,1027.7,1027.8]),colors='gray')
clabel(CS, fontsize=12, inline=1)
savefig('/home/ph290/Documents/figures/atl_meridional_cfc_density.png')
show()


'''
And how might we do some future analysis?
'''

model_future_temperature_file = '/home/ph290/Documents/teaching/canesm2_potential_temperature_rcp85_regridded.nc'

future_temperature_cube = load_cube(model_future_temperature_file)

#just do this to make sure that the cube has teh right metadata
if not future_temperature_cube.coord('latitude').has_bounds():
    future_temperature_cube.coord('latitude').guess_bounds()

if not future_temperature_cube.coord('longitude').has_bounds():
    future_temperature_cube.coord('longitude').guess_bounds()

grid_areas = iris.analysis.cartography.area_weights(future_temperature_cube)

#we'll start by having a quick look at how the heat in the ocean changes into the future in this model to get a feel for what is going on 
future_temperature_mean = future_temperature_cube.collapsed(['depth','latitude','longitude'],MEAN,weights = grid_areas)

figure()
qplt.plot(future_temperature_mean)
savefig('/home/ph290/Documents/figures/canesm2_ocean_heat.png')
show()

#It's increasing, so perhaps we would want to try and compare the first bit with observations - we will not do that here, what we will do is look at where that heat is going

#again we'll focus in on the Atlantic here
future_temperature_atlantic = future_temperature_cube.extract(atlantic_region2)
atlantic_grid_areas = iris.analysis.cartography.area_weights(future_temperature_atlantic)

#let's just start by thinking about how the heat propogates into the Atlantic, by averaging all of the latitudes and longitudes together - thsi way we only have dpth and time - which is ewaht we want
future_temperature_atlantic_with_depth = future_temperature_atlantic.collapsed(['longitude','latitude'],MEAN,weights = atlantic_grid_areas)

figure()
qplt.contourf(future_temperature_atlantic_with_depth,50)
savefig('/home/ph290/Documents/figures/canesm2_atl_heat_through_time.png')
show()

#And what about spatially? let's look at the change in teh spatial distribution of heat
#again we'll priduce and averaged slice down trhoug the ocean
future_temperature_atlantic_meridional = future_temperature_atlantic.collapsed(['longitude'],MEAN,weights = atlantic_grid_areas)

#and now we'll extract he first 20 years and the last 20 years
#this gets a bit confusing, because the time is in 'days since 1st Jan 1850'
#we can see this if we type: future_temperature_atlantic_meridional.coord('time')
#so let's pull out those values:
time_points = future_temperature_atlantic_meridional.coord('time').points

#extract first and last 20 years
future_temperature_first_20 = future_temperature_atlantic_meridional.extract(Constraint(time = time_points[0:20]))
future_temperature_last_20 = future_temperature_atlantic_meridional.extract(Constraint(time = time_points[-21:-1]))

#so lets nect average all of the yaers in each of these:
future_temperature_first_20_mean = future_temperature_first_20.collapsed('time',MEAN)
future_temperature_last_20_mean = future_temperature_last_20.collapsed('time',MEAN)

#Then we'll plot up the starting conditions and the change over the 21st century:
figure()
qplt.contourf(future_temperature_first_20_mean,50)
savefig('/home/ph290/Documents/figures/canesm2_atl_heat_first_20.png')
show()
#does ths look like the obs?


figure()
qplt.contourf(future_temperature_last_20_mean-future_temperature_first_20_mean,50)
CS=iplt.contour(future_temperature_first_20_mean,10,colors='gray')
clabel(CS, fontsize=12, inline=1)
savefig('/home/ph290/Documents/figures/canesm2_atl_heat_last20_minus_first_20.png')
show()
#how does this relate to where the heat was to start with

#Has heat and salinity changed together? DOes thsi tell us anything?
#repeat some of teh above for salinity

model_future_salinity_file = '/home/ph290/Documents/teaching/canesm2_salinity_rcp85_regridded.nc'
future_salinity_cube = load_cube(model_future_salinity_file)
if not future_salinity_cube.coord('latitude').has_bounds():
    future_salinity_cube.coord('latitude').guess_bounds()

if not future_salinity_cube.coord('longitude').has_bounds():
    future_salinity_cube.coord('longitude').guess_bounds()

future_salinity_atlantic = future_salinity_cube.extract(atlantic_region2)
salinity_atlantic_grid_areas = iris.analysis.cartography.area_weights(future_salinity_atlantic)
future_salinity_atlantic_meridional = future_salinity_atlantic.collapsed(['longitude'],MEAN,weights = salinity_atlantic_grid_areas)
time_points = future_salinity_atlantic_meridional.coord('time').points
future_salinity_first_20 = future_salinity_atlantic_meridional.extract(Constraint(time = time_points[0:20]))
future_salinity_last_20 = future_salinity_atlantic_meridional.extract(Constraint(time = time_points[-21:-1]))
future_salinity_first_20_mean = future_salinity_first_20.collapsed('time',MEAN)
future_salinity_last_20_mean = future_salinity_last_20.collapsed('time',MEAN)

temperature_change = future_temperature_last_20_mean-future_temperature_first_20_mean
salinity_change = future_salinity_last_20_mean-future_salinity_first_20_mean

figure()
scatter(temperature_change.data,salinity_change.data)
xlabel('temperature change (deg. C)')
ylabel('salinity change (psu)')
savefig('/home/ph290/Documents/figures/canesm2_atl_heat_v_salinity_scatter.png')
show()

