for corr_variable in ['sos']:
#['uas','vas','tas','hurs','sos','pr','psl','tos','evspsbl','hfss','rsntds','sic','rlds','hfls']:
    #corr_variable = 'sos'
    models_tmp = []
    composite_data = {}
    ts_variable = 'sos'
    for model in models:
                    print model
    #       try:
                    #salinity
                    cube1 = iris.load_cube(directory+model+'_'+ts_variable+'_past1000_r1i1p1*.nc')
                    cube1 = extract_years(cube1)
                    try:
                                    depths = cube1.coord('depth').points
                                    cube1 = cube1.extract(iris.Constraint(depth = np.min(depths)))
                    except:
                                    print 'no salinity depth coordinate'
                    temporary_cube = cube1.intersection(longitude = (west, east))
                    cube1_n_iceland = temporary_cube.intersection(latitude = (south, north))
                    try:
                                    cube1_n_iceland.coord('latitude').guess_bounds()
                                    cube1_n_iceland.coord('longitude').guess_bounds()
                    except:
                                    print 'already have bounds'
                    grid_areas = iris.analysis.cartography.area_weights(cube1_n_iceland)
                    cube1_n_iceland_mean = cube1_n_iceland.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=grid_areas).data
                    smoothed_data = scipy.signal.filtfilt(b, a, cube1_n_iceland_mean)
                    smoothed_data = rm.running_mean(smoothed_data,smoothing_val)
                    min_loc = np.where(smoothed_data <= scipy.stats.nanmean(smoothed_data) - scipy.stats.nanstd(smoothed_data))
                    max_loc = np.where(smoothed_data >= scipy.stats.nanmean(smoothed_data) + scipy.stats.nanstd(smoothed_data))
                    #years
                    coord = cube1_n_iceland.coord('time')
                    dt = coord.units.num2date(coord.points)
                    year_sal = np.array([coord.units.num2date(value).year for value in coord.points])
                    #pressure
                    try:
                            cube2 = iris.load_cube(directory+model+'_'+corr_variable+'_past1000_r1i1p1*.nc')
                            cube2 = extract_years(cube2)
                            try:
                                    depths = cube2.coord('depth').points
                                    cube2 = cube2.extract(iris.Constraint(depth = np.min(depths)))
                            except:
                                    print 'no variable depth coordinate'
                            cube2.data = scipy.signal.filtfilt(b, a, cube2.data,axis = 0)
                            coord = cube2.coord('time')
                            dt = coord.units.num2date(coord.points)
                            year_psl = np.array([coord.units.num2date(value).year for value in coord.points])
                            offset = 0
                            loc_min2 = np.in1d(year_psl,year_sal[min_loc]+offset)
                            loc_max2 = np.in1d(year_psl,year_sal[max_loc]+offset)
                            #output
                            composite_data[model] = {}
                            composite_data[model]['composite_high'] = cube2[loc_max2].collapsed('time',iris.analysis.MEAN)
                            composite_data[model]['composite_low'] = cube2[loc_min2].collapsed('time',iris.analysis.MEAN)
                            composite_data[model]['composite_high_minus_low'] = composite_data[model]['composite_high'] - composite_data[model]['composite_low']
                            models_tmp.append(model)
                    except:
                            print 'model can not be read in'



    for model in models_tmp:
            if np.size(np.shape(composite_data[model]['composite_high'])) == 3:
                    composite_data[model]['composite_high'] = composite_data[model]['composite_high'][0]
                    composite_data[model]['composite_low'] = composite_data[model]['composite_low'][0]


    for model in models_tmp:
         if not(corr_variable == 'msftbarot'):
            print model
            c1 = composite_data[model]['composite_low']
            c2 = composite_data[model]['composite_high']
            composite_data[model]['composite_low'].data = np.ma.masked_where(c1.data < -10000,c1.data)
            composite_data[model]['composite_high'].data = np.ma.masked_where(c2.data < -10000,c2.data)
            composite_data[model]['composite_low'].data = np.ma.masked_where(c1.data > 10000,c1.data)
            composite_data[model]['composite_high'].data = np.ma.masked_where(c2.data > 10000,c2.data)


    composite_mean_low = composite_data[models_tmp[0]]['composite_high'].copy()
    composite_mean_high = composite_data[models_tmp[0]]['composite_high'].copy()
    composite_mean_data_low = composite_mean_low.data.copy() * 0.0
    composite_mean_data_high = composite_mean_low.data.copy() * 0.0

    i = 0
    for model in models_tmp:
            i += 1
            print model
            composite_mean_data_low += composite_data[model]['composite_low'].data
            composite_mean_data_high += composite_data[model]['composite_high'].data


    composite_mean_low.data = composite_mean_data_low
    composite_mean_low = composite_mean_low / i
    composite_mean_high.data = composite_mean_data_high
    composite_mean_high = composite_mean_high / i


    # if np.size(np.shape(composite_mean_high)) == 3:
    # 	composite_mean_high = composite_mean_high[0]
    # 	composite_mean_low = composite_mean_low[0]


    min1 = np.min(composite_mean_high.data)
    min2 = np.min(composite_mean_low.data)
    min = np.min([min1,min2])
    max1 = np.max(composite_mean_high.data)
    max2 = np.max(composite_mean_low.data)
    max = np.max([max1,max2])
    max2 = np.max([max,np.abs(min)])

    if min < 0:
            min_use = min
            max_use = max



    if min >= 0:
            min_use = max2*(-1.0)
            max_use = max2	


    tmp_high = composite_mean_high.copy()
    tmp_low = composite_mean_low.copy()

    plt.close('all')
    fig = plt.figure(figsize = (20,10))
    ax1 = plt.subplot(121,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
    ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
    my_plot = iplt.contourf(tmp_high,np.linspace(min_use,max_use),cmap='bwr')
    #ax1.add_feature(cfeature.LAND,facecolor='#f6f6f6')
    plt.gca().coastlines()
    bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
    bar.set_label(corr_variable+' ('+format(tmp_high.units)+')')
    plt.title('PMIP3 high salinity yr '+corr_variable+' composites, n = '+str(i))

    ax1 = plt.subplot(122,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
    ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
    my_plot = iplt.contourf(tmp_low,np.linspace(min_use,max_use,31),cmap='bwr')
    #ax1.add_feature(cfeature.LAND,facecolor='#f6f6f6')
    plt.gca().coastlines()
    bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
    bar.set_label(corr_variable+' ('+format(tmp_low.units)+')')
    plt.title('PMIP3 low salinity yr '+corr_variable+' composites, n = '+str(i))
    plt.savefig('/home/ph290/Documents/figures/composites_'+corr_variable+'.png')
    #plt.show()


