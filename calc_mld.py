import iris
import numpy as np
import matplotlib.pyplot as plt
import glob
import iris.quickplot as qplt


#execfile('/home/ph290/Documents/python_scripts/calc_mld.py')


files = glob.glob('/data/temp/ph290/andy_w_analysis/processed/*_thetao_*.nc')

for file in files:
    model = file.split('/')[-1].split('_')[0]
    print 'processing '+model
    test = glob.glob('/data/temp/ph290/andy_w_analysis/processed/'+model+'_mld_regridded.nc')
    if np.size(test) == 0:
        try:
            cube = iris.load_cube(file)
            try:
                depths = cube.coord('depth').points
            except:
                print 'no depth'

            try:
                depths = cube.coord('generic ocean level').points
            except:
                print 'no generic ocean level'

            times =  cube.coord('time').points

            mld = cube.extract(iris.Constraint(depth = 5)).copy()
            mld.long_name = 'mixed layer depth'
            mld.units = 'm'
            mld.standard_name = 'ocean_mixed_layer_thickness'

            for k in np.arange(times.size):
                print 'month '+np.str(k)+' of '+np.str(times.size)
                x = cube[k].data
                for i in np.arange(x.shape[1]):
                    for j in np.arange(x.shape[2]):
                        tmp = np.abs(x[:,i,j]-x[-2,i,j])
                        loc = np.where(tmp[0:-2] >= 0.5)
                        if loc[0].size > 0:
                            mld.data[k,i,j] = depths[loc[0][-1]]
                # if k/10.0 == np.round(k/10.0):
                #     plt.close('all')
                #     plt.figure()
                #     qplt.contourf(mld[k])
                #     plt.show(block = False)

            iris.fileformats.netcdf.save(mld, '/data/temp/ph290/andy_w_analysis/processed/'+model+'_mld_regridded.nc','NETCDF3_CLASSIC')
        except:
            print model+' failed'
