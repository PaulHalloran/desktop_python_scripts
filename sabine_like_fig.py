import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import cartopy.crs as ccrs
import iris
import iris.analysis.cartography
import iris.plot as iplt

brewer_cmap = mpl_cm.get_cmap('brewer_OrRd_09')

file = '/home/ph290/data1/observations/glodap/AnthCO2/AnthCO2.nc'

cube = iris.load_cube(file,'Anthropogenic_CO2')
data = cube.data
#depths = cube.coord('depth').points[1:-1]-cube.coord('depth').points[0:-2]
depths = [10.,   10.,   10.,   20.,   25.,   25.,   25.,   25.,   50.,
         50.,   50.,  100.,  100.,  100.,  100.,  100.,  100.,  100.,
        100.,  100.,  100.,  100.,  100.,  250.,  250.,  500.,  500.,
        500.,  500.,  500.,  500., 500., 500.]

for i in range(33):
    data[i,:,:] = data[i,:,:] * depths[i]

cube.data = data


cube2 = cube.collapsed('depth',iris.analysis.SUM)

plt.close('all')
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
cs = iplt.contourf(cube2/1000.0,10,cmap=brewer_cmap)
ax.coastlines()
bar = plt.colorbar(cs)
bar.set_label('column total anthropogenic C (mol m$^{-2}$)')

#plt.show()
plt.savefig('/home/ph290/Documents/figures/sabine_like_fig.pdf')
