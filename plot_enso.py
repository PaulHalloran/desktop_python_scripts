# import iris
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
# import iris.coord_categorisation
# import iris.analysis.cartography
# import iris.analysis
# import scipy
# from scipy import signal
# import numpy as np
# import cartopy
import matplotlib as mpl


def set_foregroundcolor(ax, color):
     '''For the specified axes, sets the color of the frame, major ticks,                                                             
         tick labels, axis labels, title and legend                                                                                   
     '''
     for tl in ax.get_xticklines() + ax.get_yticklines():
         tl.set_color(color)
     for spine in ax.spines:
         ax.spines[spine].set_edgecolor(color)
     for tick in ax.xaxis.get_major_ticks():
         tick.label1.set_color(color)
     for tick in ax.yaxis.get_major_ticks():
         tick.label1.set_color(color)
     ax.axes.xaxis.label.set_color(color)
     ax.axes.yaxis.label.set_color(color)
     ax.axes.xaxis.get_offset_text().set_color(color)
     ax.axes.yaxis.get_offset_text().set_color(color)
     ax.axes.title.set_color(color)
     lh = ax.get_legend()
     if lh != None:
         lh.get_title().set_color(color)
         lh.legendPatch.set_edgecolor('none')
         labels = lh.get_texts()
         for lab in labels:
             lab.set_color(color)
     for tl in ax.get_xticklabels():
         tl.set_color(color)
     for tl in ax.get_yticklabels():
         tl.set_color(color)
 
 
def set_backgroundcolor(ax, color):
     '''Sets the background color of the current axes (and legend).                                                                   
         Use 'None' (with quotes) for transparent. To get transparent                                                                 
         background on saved figures, use:                                                                                            
         pp.savefig("fig1.svg", transparent=True)                                                                                     
     '''
     ax.patch.set_facecolor(color)
     lh = ax.get_legend()
     if lh != None:
         lh.legendPatch.set_facecolor(color)



# file = '/data/data1/ph290/observations/hadisst/HadISST_sst.nc'
# cube = iris.load(file)
# cube = cube[0]
# iris.coord_categorisation.add_year(cube, 'time', name='year')
# cube2 = cube.aggregated_by('year', iris.analysis.MEAN)

# lon_west = -170.0
# lon_east = -120
# lat_south = -5
# lat_north = 5

# cube.coord('latitude').guess_bounds()
# cube.coord('longitude').guess_bounds()
# grid_areas = iris.analysis.cartography.area_weights(cube)

# region = iris.Constraint(longitude=lambda v: lon_west <= v <= lon_east,latitude=lambda v: lat_south <= v <= lat_north)
# cube_region = cube2.extract(region)

# cube_region.coord('longitude').guess_bounds()
# cube_region.coord('latitude').guess_bounds()
# grid_areas_region = iris.analysis.cartography.area_weights(cube_region)

# area_avged_region = cube_region.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas_region)
# area_avged_region_detrended = area_avged_region.copy()
# area_avged_region_detrended.data = scipy.signal.detrend(area_avged_region.data)

# nino = np.where(area_avged_region_detrended.data >= 0.5)
# nina = np.where(area_avged_region_detrended.data <= 0.5)

# cube3 = cube2.copy()
# cube3.data = scipy.signal.detrend(cube2.data,axis = 0)

# nino_sst = cube3[nino].collapsed('time',iris.analysis.MEAN)
# nina_sst = cube3[nina].collapsed('time',iris.analysis.MEAN)

# plt_cube = iris.analysis.maths.subtract(nino_sst,nina_sst)
# lats = plt_cube.coord('latitude').points
# lons = plt_cube.coord('longitude').points
# lons, lats = np.meshgrid(lons, lats)


# plt.figure()
# ax = plt.axes(projection=ccrs.Orthographic(central_longitude=220.0, central_latitude=0.0))
# ax.set_global()
# CS = ax.contourf(lons,lats,plt_cube.data,np.linspace(-1.4,1.4,51),transform = ccrs.PlateCarree())
# ax.add_feature(cartopy.feature.LAND)
# ax.coastlines()
# cbar = plt.colorbar(CS)
# cbar.ax.set_xlabel('SST $^o$C')
# #plt.show()
# plt.savefig('/home/ph290/Documents/figures/enso.png',dpi = 600)

# import iris.quickplot as qplt

# mpl.rcParams['figure.figsize'] = 10, 5
# mpl.rcParams['axes.labelsize'] = 20
# mpl.rcParams['axes.labelsize'] = 20



# data = area_avged_region_detrended.data
# coord = area_avged_region_detrended.coord('time')
# dt = coord.units.num2date(coord.points)
# year = np.array([coord.units.num2date(value).year for value in coord.points])
# month  = np.array([coord.units.num2date(value).month for value in coord.points])

# fig, (ax)  = plt.subplots(nrows=1)
# ax.plot(year+month/12,data,'r',linewidth = 3)
# ax.set_xlabel('year')
# ax.set_ylabel('ENSO box SST anomaly ($^o$C)')
# ax.set_xlim(1870,2013) 
# set_foregroundcolor(ax, 'black')
# set_backgroundcolor(ax, 'white')
# ax.tick_params(axis='both', which='major', labelsize=20)
# ax.tick_params(axis='both', which='minor', labelsize=20)
# mpl.pyplot.locator_params(nbins=4)
# plt.tight_layout()
# # plt.show()                      
# plt.savefig('/home/ph290/Documents/figures/enso_ts.png', transparent=True,dpi = 500)

fig, (ax)  = plt.subplots(nrows=1)
ax.plot(year+month/12,data,'r',linewidth = 3)
ax.set_xlabel('year')
ax.set_ylabel('ENSO box SST anomaly ($^o$C)')
ax.set_xlim(1995,2000) 
set_foregroundcolor(ax, 'black')
set_backgroundcolor(ax, 'white')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
mpl.pyplot.locator_params(nbins=4)
plt.tight_layout()
plt.show()                      
# plt.savefig('/home/ph290/Documents/figures/enso_ts.png', transparent=True,dpi = 500)
