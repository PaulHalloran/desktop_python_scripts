import numpy as np
import matplotlib.pyplot as plt
import iris
import iris.analysis.cartography
import iris.analysis
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt

def my_callback(cube, field, filename):
    if field.lbuser[3] != 101:
        raise iris.exceptions.IgnoreCubeException()

def pp_file_volume_total(filename,stash,lon1,lat1,lon2,lat2,depth_m):
    stash_tmp=[]
    midpoint_depth=[]
    bottom_depth=[]
    #
    for field in iris.fileformats.pp.load(filename):
        stash_tmp.append(field.lbuser[3])
        midpoint_depth.append(field.blev)
        bottom_depth.append(field.brlev)
    #
    bottom_depth=np.array(bottom_depth)
    midpoint_depth=np.array(midpoint_depth)
    loc=np.where(np.array(stash_tmp) == stash)
    layer_thickness = (bottom_depth[loc]-midpoint_depth[loc])*2.0
    #
    cube = iris.load(filename, callback=my_callback)
    cube[0].data *= 0.0
    cube[0].data += 1.0
    #
    region = iris.Constraint(longitude=lambda v: lon1 <= v <= lon2,latitude=lambda v: lat1 <= v <= lat2,depth=lambda v: 0 <= v <= depth_m)
    #
    cube_region=cube[0].extract(region)
    cube_region.coord('latitude').guess_bounds()
    cube_region.coord('longitude').guess_bounds()
    weights = iris.analysis.cartography.area_weights(cube_region[0])
    #np.sum(weights) # area of earth in m2
    #
    layer_vol=np.empty(np.size(cube_region.coord('depth').points))
    for i in range(np.size(cube_region.coord('depth').points)):
        area=cube_region.extract(iris.Constraint(model_level_number = i+1)).data*weights
        total_area=np.sum(area)
        layer_vol[i]=(total_area*layer_thickness[i])

    total_vol=np.sum(layer_vol)

    return total_vol


filename='/project/obgc/mass_retrievals/hadcm3_files/akinao.pyd7c10.pp'
stash=101

cube = iris.load(filename, callback=my_callback)

lon1=-60+360
lon2=20+360
lat1=-30
lat2=48
depth_m=1000.0

region = iris.Constraint(longitude=lambda v: lon1 <= v <= lon2,latitude=lambda v: lat1 <= v <= lat2,depth=lambda v: 0 <= v <= 1000)
cube_region=cube[0].extract(region)
qplt.contourf(cube_region[0], 25)
plt.gca().coastlines()
plt.show()

ocean_volume_m3=pp_file_volume_total(filename,stash,lon1,lat1,lon2,lat2,depth_m)
print ocean_volume_m3
