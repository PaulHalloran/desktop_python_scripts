'''
To be run after palaeo_amo_amoc_paper_figures_iv.py
'''
smoothing_val = 10

solar2 = np.genfromtxt('/home/ph290/data0/misc_data/last_millenium_solar/tsi_VK.txt',skip_header = 4)
solar2_yr = solar2[:,0]
loc = np.where((solar2_yr > 850) & (solar2_yr < 1850))[0]
solar2_yr = solar2_yr[loc]
solar2_data = solar2[loc,1]
solar2_data = scipy.signal.filtfilt(b, a, solar2_data)
solar2_data = rm.running_mean(solar2_data,smoothing_val)

solar3 = np.genfromtxt('/home/ph290/data0/misc_data/last_millenium_solar/tsi_MEA_11yr.txt',skip_header = 4)
solar3_yr = solar3[:,0]
loc = np.where((solar3_yr > 850) & (solar3_yr < 1850))[0]
solar3_yr = solar3_yr[loc]
solar3_data = solar3[loc,2]
solar3_data = scipy.signal.filtfilt(b, a, solar3_data)
solar3_data = rm.running_mean(solar3_data,smoothing_val)

solar4 = np.genfromtxt('/home/ph290/data0/misc_data/last_millenium_solar/tsi_DB_lin_40_11yr.txt',skip_header = 4)
solar4_yr = solar4[:,0]
loc = np.where((solar4_yr > 850) & (solar4_yr < 1850))[0]
solar4_yr = solar4_yr[loc]
solar4_data = solar4[loc,2]
solar4_data = scipy.signal.filtfilt(b, a, solar4_data)
solar4_data = rm.running_mean(solar4_data,smoothing_val)

solar = np.empty([np.size(solar2_yr),3])
solar[:,0] = solar2_data
solar[:,1] = solar3_data
solar[:,2] = solar4_data
solar = np.mean(solar,axis = 1)
solar_yr = solar2_yr

import string
names = list(string.ascii_lowercase)

offsets = np.array([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
for j,offset in enumerate(offsets):
    models_tmp = []
    composite_data = {}
    ts_variable = vol_globe
    corr_variable = 'sos'
    smoothing_val = 1
    for model in models:
        print model
        smoothed_data = solar
        cutoff_1 = 0.1
        cutoff_2 = -0.1
        #np.median(smoothed_data)
        min_loc = np.where(smoothed_data <= cutoff_2)
        max_loc = np.where(smoothed_data >= cutoff_1)
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
                year_corr_variable = np.array([coord.units.num2date(value).year for value in coord.points])
                loc_min2 = np.in1d(year_corr_variable,solar_yr[min_loc]+offset)
                loc_max2 = np.in1d(year_corr_variable,solar_yr[max_loc]+offset)
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

    min_use = -0.05
    max_use = 0.05

    tmp_high = composite_mean_high.copy()
    tmp_low = composite_mean_low.copy()

    plt.close('all')
    fig = plt.figure(figsize = (10,10))
    ax1 = plt.subplot(111,projection=ccrs.NorthPolarStereo(central_longitude=0.0))
    ax1.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
    my_plot = iplt.contourf(tmp_high,np.linspace(min_use,max_use),cmap='bwr')
    #ax1.add_feature(cfeature.LAND,facecolor='#f6f6f6')
    plt.gca().coastlines()
    bar = plt.colorbar(my_plot, orientation='horizontal', extend='both')
    bar.set_label(corr_variable+' ('+format(tmp_high.units)+')')
    plt.title('PMIP3 high salinity yr '+corr_variable+' composites, n = '+str(i))
    plt.savefig('/home/ph290/Documents/figures/low_variability_solar_composites_'+corr_variable+'_'+names[j]+'.png')
    #plt.show()


