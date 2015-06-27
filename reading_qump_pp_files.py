import iris
import iris.analysis.cartography
import iris.analysis
import numpy as np
import iris.coords as coords
import matplotlib.pyplot as plt
import glob
from iris.fileformats.pp import STASH

# c_stash_249=iris.AttributeConstraint(STASH='m02s30i249')
# c_stash_200=iris.AttributeConstraint(STASH='m02s00i200')
# c_stash_101=iris.AttributeConstraint(STASH='m02s00i101')
# c_stash_102=iris.AttributeConstraint(STASH='m02s00i102')
# c_stash_103=iris.AttributeConstraint(STASH='m02s00i103')
# c_stash_104=iris.AttributeConstraint(STASH='m02s00i104')
# c_stash_321=iris.AttributeConstraint(STASH='m02s30i321')

# sp_atl_region = iris.Constraint(longitude=lambda v: -60+360 <= v <= -10+360,latitude=lambda v: 48 <= v <= 65)
# tr_atl_region = iris.Constraint(longitude=lambda v: -60+360 <= v <= -10+360,latitude=lambda v: 30 <= v <= 48)
# s_atl_region = iris.Constraint(longitude=lambda v: -60+360 <= v <= -10+360,latitude=lambda v: -90 <= v <= -30)
# big_region = iris.Constraint(longitude=lambda v: -60+360 <= v <= -10+360,latitude=lambda v: -60 <= v <= 65)

# regions=[sp_atl_region,tr_atl_region,s_atl_region,big_region]

# dir='/project/obgc/qump_n_atl_mor_var_monthly/'

# in_file='/home/h04/hador/qump_n_atl/qump_run_names.txt'
# run_ids = np.genfromtxt(in_file,skip_header=1,usecols=[2,5,7],delimiter=",",dtype=list)
# run_ids_shape=run_ids.shape

# #read in one file to work our grid weights before entering big loop...
# filenames=glob.glob(dir+run_ids[i,0]+'/*.pp')
# cube = iris.load(filenames[0])

# stash_101_cube = cube.extract(c_stash_101)
# stash_101_cube = stash_101_cube[0]

# stash_101_sp_cube = stash_101_cube.extract(sp_atl_region)
# stash_101_tr_cube = stash_101_cube.extract(tr_atl_region)
# stash_101_s_cube = stash_101_cube.extract(s_atl_region)
# stash_101_big_cube = stash_101_cube.extract(big_region)

# stash_101_sp_cube.coord('latitude').guess_bounds()
# stash_101_sp_cube.coord('longitude').guess_bounds()

# stash_101_tr_cube.coord('latitude').guess_bounds()
# stash_101_tr_cube.coord('longitude').guess_bounds()

# stash_101_s_cube.coord('latitude').guess_bounds()
# stash_101_s_cube.coord('longitude').guess_bounds()

# stash_101_big_cube.coord('latitude').guess_bounds()
# stash_101_big_cube.coord('longitude').guess_bounds()

# grid_areas_sp = iris.analysis.cartography.area_weights(stash_101_sp_cube)
# grid_areas_tr =  iris.analysis.cartography.area_weights(stash_101_tr_cube)
# grid_areas_s = iris.analysis.cartography.area_weights(stash_101_s_cube)
# grid_areas_big = iris.analysis.cartography.area_weights(stash_101_big_cube)

# grid_areas=[grid_areas_sp,grid_areas_tr,grid_areas_s,grid_areas_big]

# stash_names=['249','200','101','102','103','104']
x=1
#for i in range(run_ids_shape[0]):
#REMOVE!!!!!!!!!!!!!!!
while x == 1:
    i=5
    print 'i = '+str(i)
    filenames=[]
    for f in glob.glob(dir+run_ids[i,0]+'*/*.*.pp'):
        filenames.append(f)
    for f in glob.glob(dir+run_ids[i,1]+'/*.*.pp'):
        filenames.append(f)
    for f in glob.glob(dir+run_ids[i,2]+'/*,*.pp'):
        filenames.append(f)

    print 'loading cube'
    cube = iris.load(filenames)
    print 'cube loaded'
    cube2=cube[0]

    stash_249_cube = cube.extract(c_stash_249)
    stash_249_cube = stash_249_cube[0]

    stash_200_cube = cube.extract(c_stash_200)
    stash_200_cube = stash_200_cube[0]

    stash_101_cube = cube.extract(c_stash_101)
    stash_101_cube = stash_101_cube[0]

    stash_102_cube = cube.extract(c_stash_102)
    stash_102_cube = stash_102_cube[0]

    stash_103_cube = cube.extract(c_stash_103)
    stash_103_cube = stash_103_cube[0]

    stash_104_cube = cube.extract(c_stash_104)
    stash_104_cube = stash_104_cube[0]

    stash_321_cube = cube.extract(c_stash_321)
    stash_321_cube = stash_321_cube[0]

    all_stashs=[stash_249_cube,stash_200_cube,stash_101_cube,stash_102_cube,stash_103_cube,stash_104_cube]

    for j,dummy in enumerate(all_stashs):
        print 'extracting and meaning stash '+str(j)+' of '+str(np.size(all_stashs))
        #for k,dummy2 in enumerate(regions):
        sp_mean=all_stashs[j].extract(regions[0]).collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=np.array([grid_areas[0]]*cube2.coord("time").points.size))
        tr_mean=all_stashs[j].extract(regions[1]).collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=np.array([grid_areas[1]]*cube2.coord("time").points.size))
        s_mean=all_stashs[j].extract(regions[2]).collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=np.array([grid_areas[2]]*cube2.coord("time").points.size))
        big_mean=all_stashs[j].extract(regions[3]).collapsed(['longitude','latitude'], iris.analysis.MEAN,weights=np.array([grid_areas[3]]*cube2.coord("time").points.size))
        years = (sp_mean.coord("time").points/24.0/360.0)+1970.0
        
        np.savetxt('/project/obgc/qump_out_python/qump_data_run_'+run_ids[i,0]+'_stash_'+stash_names[j]+'.txt', np.vstack((years,sp_mean.data,tr_mean.data,s_mean.data,big_mean.data)).T, delimiter=',')
            
    #REMOVE!!!!!!!!!!!!!!!
    x=2
