import numpy as np
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis
import iris.coords as icoords
import cartopy.crs as ccrs
import cartopy



directory='/Users/ph290/Public/mo_data/nemo_shelf/'
filename = directory + '20120101__amm7.25hourm.grid_T.nc'

cube_in = iris.load_cube(filename,'Net Primary Production')
cube = cube_in.collapsed(['model_level_number','time'], iris.analysis.SUM)

#existing records
no_records=6
xs=np.empty(no_records)
ys=np.empty(no_records)
#irish sea
xs[0]=-4.5
ys[0]=54.1
#north sea
xs[1]=-0.205
ys[1]=58.5
#north sea II
xs[2]=0.9
ys[2]= 59.7
#north sea III
xs[3]=0.2
ys[3]= 58.47
#north sea IV
xs[4]=0.31
ys[4]=59.2
#scotland
xs[5]=-6.24
ys[5]=56.4

#lots of shells, no records yet
no_records2=5
xs2=np.empty(no_records2)
ys2=np.empty(no_records2)
#Belfast
xs2[0]=-5.35
ys2[0]=54.42
#East coast of Northern Ireland
xs2[1]=-5.25
ys2[1]=54.21
#Caernarfon Bay
xs2[2]=-4.35
ys2[2]=53.3
#Cardigan Bay
xs2[3]=-4.19
ys2[3]=52.2
#Celtic Sea
xs2[4]=-6.7
ys2[4]=51.45


    
lo=0
hi=30

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
data=cube.data
lons = cube.coord('longitude').points
lats = cube.coord('latitude').points
cube_label = 'latitude: %s' % cube.coord('latitude').points
levels = np.linspace(lo,hi,20)
data2=np.array(data)
data2[np.where(data2 > hi)]=hi
data2[np.where(data2 == 0)] = np.nan
data2[np.where(data2 < lo)]=lo
contour=plt.contourf(lons, lats, data2,levels=levels,cmap='jet',xlabel=cube_label)
cartopy.feature.LAND.scale='50m'
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.RIVERS)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.coastlines(resolution='50m')
cbar = plt.colorbar(contour, ticks=np.linspace(lo,hi,7), orientation='horizontal')
cbar.set_label('Net Primary Production (mg C m$^{-2}$)')
# enable axis ticks
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

plt.plot(xs, ys,
         color='red', linewidth=0, marker='D',
         transform=ccrs.PlateCarree(),
         )

plt.plot(xs2, ys2,
         color='red', linewidth=0, marker='o',
         transform=ccrs.PlateCarree(),
         )

plt.show()


#     plt.savefig('/home/h04/hador/public_html/twiki_figures/carib_sst_and_stdev.png')


