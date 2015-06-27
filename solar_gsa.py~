'''
To be run after palaeo_amo_amoc_paper_figures_iv.py
'''

file1 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090N_AOD_c.txt'
file2 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030N_AOD_c.txt'
file3 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_3090S_AOD_c.txt'
file4 = '/data/data0/ph290/misc_data/last_millenium_volcanic/ICI5_030S_AOD_c.txt'

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)
data4 = np.genfromtxt(file4)

data_tmp = np.zeros([data1.shape[0],2])
data_tmp[:,0] = data1[:,1]
data_tmp[:,1] = data2[:,1]
data = np.mean(data_tmp,axis = 1)
voln_n = data1.copy()
voln_n[:,1] = data

data_tmp[:,0] = data3[:,1]
data_tmp[:,1] = data4[:,1]
data = np.mean(data_tmp,axis = 1)
voln_s = data1.copy()
voln_s[:,1] = data

data_tmp[:,0] = data2[:,1]
data_tmp[:,1] = data4[:,1]
data = np.mean(data_tmp,axis = 1)
vol_eq = data1.copy()
vol_eq[:,1] = data

data_tmp = np.zeros([data1.shape[0],4])
data_tmp[:,0] = data2[:,1]
data_tmp[:,1] = data4[:,1]
data_tmp[:,2] = data1[:,1]
data_tmp[:,3] = data3[:,1]
data = np.mean(data_tmp,axis = 1)
vol_globe = data1.copy()
vol_globe[:,1] = data

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
        volc_year = voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,0]
        volc_year = np.floor(volc_year)
        unique_yr = np.unique(volc_year)
        volc_year2 = unique_yr
        volc = voln_n[(start_year-voln_n[0,0])*36:(end_year-voln_n[0,0])*36,1]
        volc2 = unique_yr.copy()
        for i,yr in enumerate(unique_yr):
            loc = np.where(volc_year == yr)
            volc2[i] = np.mean(volc[loc])
        smoothed_data = running_mean_post.running_mean_post(volc2,smoothing_val)
        cutoff_1 = 0.01
        #np.median(smoothed_data)
        cuttoff_2 = 0.01
        min_loc = np.where(smoothed_data <= cutoff_1)
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
                loc_min2 = np.in1d(year_corr_variable,volc_year2[min_loc]+offset)
                loc_max2 = np.in1d(year_corr_variable,volc_year2[max_loc]+offset)
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
    plt.savefig('/home/ph290/Documents/figures/volc_composites_'+corr_variable+'_'+names[j]+'.png')
    #plt.show()


