import glob
import numpy as np
import iris
import matplotlib.pyplot as plt
import iris.quickplot as qplt

files = glob.glob('/media/usb_external1/cmip5/thetao/regridded/*.nc')

names = []
starts = []
ends = []

starts_layer = []
ends_layer = []

for file in files:
    print file.split('/')[-1].split('_')[0]
    names.append(file.split('/')[-1].split('_')[0])
    cube = iris.load_cube(file)
    coord_names = [coord.name() for coord in cube.coords()]
    test1 = np.size(np.where(np.array(coord_names) == 'ocean sigma coordinate'))
    test1b = np.size(np.where(np.array(coord_names) == 'ocean sigma over z coordinate'))
    if test1 == 1:
            cube.coord('ocean sigma coordinate').long_name = 'depth'
    if test1b == 1:
            cube.coord('ocean sigma over z coordinate').long_name = 'depth'
    starts.append(cube[0:20].collapsed(['longitude','time'],iris.analysis.MEAN))
    ends.append(cube[-20:-1].collapsed(['longitude','time'],iris.analysis.MEAN))
    tmp = cube[0:20].collapsed(['time'],iris.analysis.MEAN)
    starts_layer.append(tmp.extract(iris.Constraint(depth = 500)))
    tmp = cube[-20:-1].collapsed(['time'],iris.analysis.MEAN)
    ends_layer.append(tmp.extract(iris.Constraint(depth = 500)))

for i,name in enumerate(names):
    print name
    if (name in 'ACCESS1-3') or (name in 'BNU-ESM') or (name in 'MIROC5'):
        print 'removed'
        starts.pop(i)
        ends.pop(i)
        names.pop(i)
        ends_layer.pop(i)
        starts_layer.pop(i)

# for i in np.arange(np.size(names)):
#     plt.close()
#     plt.figure
#     qplt.contourf(ends[i]-starts[i],np.linspace(-1,5,51))
#     plt.title(names[i])
#     plt.savefig('/home/ph290/Documents/figures/thetao/'+names[i]+'.png',dpi = 500)
#     plt.close()

diff = np.array(ends)-np.array(starts)
diff_layer = np.array(ends_layer)[1:]-np.array(starts_layer)[1:]
plt.close()

plt.figure()
mean_cube = np.mean(diff)
qplt.contourf(mean_cube,51)
plt.title('mean')
plt.savefig('/home/ph290/Documents/figures/thetao/mean.png',transparent = True,dpi = 500)
#plt.show(block = False)

plt.figure()
var_cube = np.var(diff)
qplt.contourf(var_cube,51)
plt.title('variance')
plt.savefig('/home/ph290/Documents/figures/thetao/var.png',transparent = True,dpi = 500)
#plt.show(block = False)


plt.figure()
mean_cube_layer = np.mean(diff_layer)
qplt.contourf(mean_cube_layer,51)
plt.gca().coastlines()
plt.title('mean')
plt.savefig('/home/ph290/Documents/figures/thetao/mean_layer.png',transparent = True,dpi = 500)
#plt.show(block = False)

plt.figure()
var_cube_layer = np.var(diff_layer)
qplt.contourf(var_cube_layer,51)
plt.gca().coastlines()
plt.title('variance')
plt.savefig('/home/ph290/Documents/figures/thetao/var_layer.png',transparent = True,dpi = 500)
#plt.show(block = False)

