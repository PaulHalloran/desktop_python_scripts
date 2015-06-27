plt.close('all')
fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

l1 = ax1.plot(np.arange(850,1851),tas_n_iceland_mean,'b',linewidth = 2,alpha = 0.75,label = 'GIN Sea air temperature')
#ax1b = ax1.twinx()
x = meaned_solar2*0.2 - rmp.running_mean_post(meaned_volc2,7)
x = scipy.signal.filtfilt(b1, a1, x,axis = 0)+0.55
x2 = meaned_solar2*0.2
x2 = scipy.signal.filtfilt(b1, a1, x2,axis = 0)+0.55
#ax1b.plot(np.arange(850,1851),x,'r',linewidth = 2,alpha = 0.75)
#ax1b.plot(np.arange(850,1851),x2,'g',linewidth = 2,alpha = 0.75)
l2 = ax1.plot(np.arange(850,1851),x,'r',linewidth = 2,alpha = 0.75,label = 'Simple model of air temperature using volcanic and solar index')
l3 = ax1.plot(np.arange(850,1851),x2,'g',linewidth = 2,alpha = 0.75,label = 'Simple model of air temperature using just solar index')
ax1.set_xlabel('calendar year')
ax1.set_ylabel('temperature ($^{\circ}$C)')
ax1.set_xlim([950,1850])

lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,loc = 'lower left', fancybox=True, framealpha=0.2,prop={'size':12})


brewer_cmap = 'bwr'
levels = np.linspace(-0.2,0.2,31)

ax2 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, projection = ccrs.PlateCarree())
ax3 = plt.subplot2grid((3, 2), (0, 1), rowspan=2, projection = ccrs.PlateCarree())

to_plot = volc_composite_mean_high-volc_composite_mean_low
lons = to_plot.coord('longitude').points
lats = to_plot.coord('latitude').points
# ax1 = plt.subplot(121, projection = ccrs.PlateCarree())
ax2.set_extent((-90.0, 20.0, 0.0, 90.0), crs=ccrs.PlateCarree())
contour = ax2.contourf(lons, lats, to_plot.data,levels=levels,cmap=brewer_cmap)
cartopy.feature.LAND.scale='50m'
ax2.add_feature(cartopy.feature.LAND)
ax2.coastlines(resolution='50m')
ax2.set_title('Composite of high minus\nlow volcanic years')

to_plot = solar_composite_mean_low - solar_composite_mean_high
lons = to_plot.coord('longitude').points
lats = to_plot.coord('latitude').points
# ax2 = plt.subplot(122, projection = ccrs.PlateCarree())
ax3.set_extent((-90.0, 20.0, 0.0, 90.0), crs=ccrs.PlateCarree())
contour = ax3.contourf(lons, lats, to_plot.data,levels=levels,cmap=brewer_cmap)
cartopy.feature.LAND.scale='50m'
ax3.add_feature(cartopy.feature.LAND)
ax3.coastlines(resolution='50m')
ax3.set_title('Composite of high minus\nlow solar years')


cax = fig.add_axes([0.1, 0.45, 0.85, 0.03])

cbar = plt.colorbar(contour, cax=cax, ticks=[-0.2, -0.1,0, 0.1,0.2],orientation = 'horizontal')
cbar.set_label('temperature anomaly ($^{\circ}$C)')

plt.tight_layout()


plt.show()
# plt.savefig('/home/ph290/Documents/figures/volc_sol_tas_comp.png')

