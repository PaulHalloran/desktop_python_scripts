import iris
import iris.analysis.cartography
import iris.analysis
import numpy as np
import iris.coords as coords
import matplotlib.pyplot as plt
import glob
from iris.fileformats.pp import STASH
import math

def my_callback(cube, field, filename):
        cube.remove_coord('forecast_reference_time')
        cube.remove_coord('forecast_period')
        #the cubes were not merging properly before, because the time coordinate appeard to have teo different names... I think this may work

def distance_on_unit_sphere(lat1, long1, lat2, long2):
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0       
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    # Compute spherical distance from spherical coordinates.
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return arc*6373000.0

c_stash_249=iris.AttributeConstraint(STASH='m02s30i249')
c_stash_200=iris.AttributeConstraint(STASH='m02s00i200')
c_stash_101=iris.AttributeConstraint(STASH='m02s00i101')
c_stash_102=iris.AttributeConstraint(STASH='m02s00i102')
c_stash_103=iris.AttributeConstraint(STASH='m02s00i103')
c_stash_104=iris.AttributeConstraint(STASH='m02s00i104')
c_stash_321=iris.AttributeConstraint(STASH='m02s30i321')

sp_atl_region = iris.Constraint(longitude=lambda v: -60+360 <= v <= -10+360,latitude=lambda v: 48 <= v <= 65)
tr_atl_region = iris.Constraint(longitude=lambda v: -60+360 <= v <= -10+360,latitude=lambda v: 30 <= v <= 48)
s_atl_region = iris.Constraint(longitude=lambda v: -60+360 <= v <= -10+360,latitude=lambda v: -90 <= v <= -30)
big_region = iris.Constraint(longitude=lambda v: -60+360 <= v <= -10+360,latitude=lambda v: -60 <= v <= 65)

regions=[sp_atl_region,tr_atl_region,s_atl_region,big_region]

dir='/data/local2/qump_n_atl_mor_var_monthly/'

in_file='/home/h04/hador/qump_n_atl/qump_run_names.txt'
run_ids = np.genfromtxt(in_file,skip_header=1,usecols=[2,5,7],delimiter=",",dtype=list)
run_ids_shape=run_ids.shape

#read in one file to work our grid weights before entering big loop...
i=0
filenames=glob.glob(dir+run_ids[i,0]+'/*.*.pp')
cube = iris.load(filenames[0])

stash_101_cube = cube.extract(c_stash_101)
stash_101_cube = stash_101_cube[0]

stash_101_sp_cube = stash_101_cube.extract(sp_atl_region)
stash_101_tr_cube = stash_101_cube.extract(tr_atl_region)
stash_101_s_cube = stash_101_cube.extract(s_atl_region)
stash_101_big_cube = stash_101_cube.extract(big_region)

stash_101_sp_cube.coord('latitude').guess_bounds()
stash_101_sp_cube.coord('longitude').guess_bounds()

stash_101_tr_cube.coord('latitude').guess_bounds()
stash_101_tr_cube.coord('longitude').guess_bounds()

stash_101_s_cube.coord('latitude').guess_bounds()
stash_101_s_cube.coord('longitude').guess_bounds()

stash_101_big_cube.coord('latitude').guess_bounds()
stash_101_big_cube.coord('longitude').guess_bounds()

grid_areas_sp = iris.analysis.cartography.area_weights(stash_101_sp_cube)
grid_areas_tr =  iris.analysis.cartography.area_weights(stash_101_tr_cube)
grid_areas_s = iris.analysis.cartography.area_weights(stash_101_s_cube)
grid_areas_big = iris.analysis.cartography.area_weights(stash_101_big_cube)

grid_areas=[grid_areas_sp,grid_areas_tr,grid_areas_s,grid_areas_big]

stash_names=['249','200','101','102','103','104']


#work out background data for stream function calculation (cell widths and thicknesses)

filenames=[]
for f in glob.glob(dir+run_ids[0,0]+'*/*.*.pp'):
    filenames.append(f)

for f in glob.glob(dir+run_ids[0,1]+'/*.*.pp'):
    filenames.append(f)

for f in glob.glob(dir+run_ids[0,2]+'/*.*.pp'):
    filenames.append(f)

stash_321_cubes = iris.load(filenames[0],iris.AttributeConstraint(STASH='m02s30i321'),callback=my_callback)
stash_321_cube=stash_321_cubes[0]

if not stash_321_cube.coord('latitude').has_bounds():
    stash_321_cube.coord('latitude').guess_bounds()
    stash_321_cube.coord('longitude').guess_bounds()

latitudes=stash_321_cube[0].coord('latitude').points
extract_latitudes=latitudes[latitudes >= 30]

cube_slice=stash_321_cube.extract(iris.Constraint(latitude=extract_latitudes[0]))
cube_slice2=cube_slice.extract(iris.Constraint(longitude = lambda v: 360-90 < v < 360))

if not cube_slice2.coord('latitude').has_bounds():
    cube_slice2.coord('longitude').guess_bounds()

longitude_bounds=cube_slice2[0].coord('longitude').bounds
cell_width_m=[]
for i,x in enumerate(longitude_bounds[:,0]):
    cell_width_m.append(distance_on_unit_sphere(extract_latitudes[0], longitude_bounds[i,0], extract_latitudes[0], longitude_bounds[i,1]))

stash_tmp=[]
midpoint_depth=[]
bottom_depth=[]

for field in iris.fileformats.pp.load(filenames[0]):
    stash_tmp.append(field.lbuser[3])
    midpoint_depth.append(field.blev)
    bottom_depth.append(field.brlev)

bottom_depth=np.array(bottom_depth)
midpoint_depth=np.array(midpoint_depth)
loc=np.where(np.array(stash_tmp) == 30321)
layer_thickness = (bottom_depth[loc]-midpoint_depth[loc])*2.0

array_shape=cube_slice2.shape
thickness_array=np.transpose([layer_thickness]*array_shape[1])
width_array=np.array([cell_width_m]*array_shape[0])


for i in range(run_ids_shape[0]):
#    if i < 26:
#        continue
        #breaks out of the loop if the counter is less than 5 to skip over the ones which have already been processed

print 'processing '+str(i)+' out of '+str(np.size(run_ids[:,0]))

filenames=[]

for f in glob.glob(dir+run_ids[i,0]+'/*.*.pp'):
    filenames.append(f)

for f in glob.glob(dir+run_ids[i,1]+'/*.*.pp'):
    filenames.append(f)

for f in glob.glob(dir+run_ids[i,2]+'/*.*.pp'):
    filenames.append(f)

print 'loading cubes (takes a while)'
cube = iris.load(np.unique(filenames),callback=my_callback)
#by introduceing the callback here can I get around reading the cube in twice?
print 'cube loaded'

cube2=cube[0]

stash_249_cubes = cube.extract(c_stash_249)
stash_249_cube = stash_249_cubes[0]

stash_200_cubes = cube.extract(c_stash_200)
stash_200_cube = stash_200_cubes[0]

stash_101_cubes = cube.extract(c_stash_101)
stash_101_cube = stash_101_cubes[0]

stash_102_cubes = cube.extract(c_stash_102)
stash_102_cube = stash_102_cubes[0]

stash_103_cubes = cube.extract(c_stash_103)
stash_103_cube = stash_103_cubes[0]

stash_104_cubes = cube.extract(c_stash_104)
stash_104_cube = stash_104_cubes[0]

stash_321_cubesb = cube.extract(c_stash_321)
stash_321_cube = stash_321_cubesb[0]

s249_size=stash_249_cube.coord('time').points.size
s200_size=stash_200_cube.coord('time').points.size
s101_size=stash_101_cube.coord('time').points.size
s102_size=stash_102_cube.coord('time').points.size
s103_size=stash_103_cube.coord('time').points.size
s104_size=stash_104_cube.coord('time').points.size
s321_size=stash_321_cube.coord('time').points.size

dates1=stash_104_cube.coord('time').points
dates2=stash_101_cube.coord('time').points
np.size(dates1)
np.size(dates2)
x=list(set(dates2) - set(dates1))
np.size(x)
print np.size(x)+np.size(dates1)
print np.sort((np.array(x)/24.0/360.0)+1970.0+1.0/360.0)

    if s249_size == s200_size == s101_size == s102_size == s103_size == s104_size == s321_size:
        #it appears that this criteria skips over all of the files with corrupted alkalinity - which is good (but we do loose a lot fo runs)
        #example: xconv -i /net/project/obgc/qump_n_atl_mor_var_monthly/aldpk/aldpko.pmt9apr.pp
        #it is possible it is skpping over other issues, but hopefully now - can check
        #actually, it might be that the unpacking process, when I unpacked them from the basic mass download (not sure why they were not unpacked on the way) may have done this - 'cos for at least some o fthe occurances, the corrupted fields do not exist in the unpacked dataset.
            
        if not stash_321_cube.coord('latitude').has_bounds():
            stash_321_cube.coord('latitude').guess_bounds()
            stash_321_cube.coord('longitude').guess_bounds()

        cube_slice=stash_321_cube.extract(iris.Constraint(latitude=extract_latitudes[0]))
        cube_slice2=cube_slice.extract(iris.Constraint(longitude = lambda v: 360-90 < v < 360))

        cube_shape=cube_slice2.shape
        dates=(cube_slice2.coord('time').points/24.0/360.0)+1970.0+1.0/360.0

        moc_stream_fun=[]
        print 'calculating moc stream function for: ',
        for count,sub_cube in enumerate(cube_slice2.slices(['model_level_number','longitude'])):
            print ' '+str(dates[count]),
            #loop through the months and year
            stream_fun=(sub_cube.data/100.0*thickness_array*width_array).sum(axis=1).cumsum(axis=0)
            #stream function is calculated:
            # northwards velocity (m) * cell width (m) * cell thickness (m). Summed across a latitude mand over basin, then integrated vertically - then the max. value taken.
            # v-velocity(cm) neds to be converted to meters.
            moc_stream_fun.append(np.max(stream_fun)/1.0e6)

        #print moc_stream_fun


        all_stashs=[stash_249_cube,stash_200_cube,stash_101_cube,stash_102_cube,stash_103_cube,stash_104_cube]
        stash_names=['30249','200','101','102','103','104']

        for j,dummy in enumerate(all_stashs):
            print 'extracting and meaning stash '+stash_names[j]
            #for k,dummy2 in enumerate(regions):
            temp=all_stashs[j].extract(regions[0])
            temp_size=temp.coord("time").points.size
            sp_mean=all_stashs[j].extract(regions[0]).collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=np.array([grid_areas[0]]*temp_size))
            tr_mean=all_stashs[j].extract(regions[1]).collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=np.array([grid_areas[1]]*temp_size))
            s_mean=all_stashs[j].extract(regions[2]).collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=np.array([grid_areas[2]]*temp_size))
            big_mean=all_stashs[j].extract(regions[3]).collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=np.array([grid_areas[3]]*temp_size))
            years = (sp_mean.coord("time").points/24.0/360.0)+1970.0+1.0/360.0
            #stash_249_cube.coord('time').units
            np.savetxt('/project/obgc/qump_out_python/qump_data_run_'+run_ids[i,0]+'_stash_'+stash_names[j]+'.txt', np.vstack((years,sp_mean.data,tr_mean.data,s_mean.data,big_mean.data)).T, delimiter=',')

        np.savetxt('/project/obgc/qump_out_python/qump_data_run_'+run_ids[i,0]+'_moc_stm_fun.txt', np.vstack((years,moc_stream_fun)).T, delimiter=',')
            
