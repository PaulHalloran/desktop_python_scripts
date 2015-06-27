import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import iris
import glob

dirctory = '/home/ph290/data1/aerosol_data/'

file_names = glob.glob(dirctory+'*.nc')

file_start_year=[]

for tmp in file_names:
	tmp2 = tmp.split('_')
	tmp3 = tmp2[-1].split('-')
	file_start_year.append(tmp3[0])

file_start_year = np.array(file_start_year)
file_start_year = file_start_year.astype(np.float)

loc=np.where(file_start_year < 2000)

hist_years = np.sort(file_start_year[loc])

for yr,i in enumerate(hist_years):
	loc = np.where(file_start_year = yr)
	cube=iris.load(file_names[loc])


# def plotting(var_number):
# 	lns=[]
# 	x=np.arange(np.size(files))
# 	for i in x[::-1]:
# 		ln=plt.plot(scenarios[i][:,0],scenarios[i][:,var_number],linewidth=3,label=names[i])
# 		lns += ln
		
# 	plt.xlabel('year')	
# 	plt.ylabel(variable[var_number]+' ('+units[var_number]+')')	

# 	labs=[l.get_label() for l in lns]
# 	plt.legend(lns,labs,loc=0).draw_frame(False)
# 	plt.tight_layout()
# 	plt.show()

# def plotting2(var_number):
# 	lns=[]
# 	x=np.arange(np.size(files))
# 	for i in x[::-1]:
# 		ln=plt.plot(scenarios[i][:,0],scenarios[i][:,var_number],linewidth=3,label=names[i])
# 		lns += ln
		
# 	plt.xlim(1950,2100)
# 	plt.ylim(250,1000)
# 	plt.xlabel('year')	
# 	plt.ylabel(variable[var_number]+' ('+units[var_number]+')')	

# 	labs=[l.get_label() for l in lns]
# 	plt.legend(lns,labs,loc=0).draw_frame(False)
# 	plt.tight_layout()
# 	plt.show()

# units = ['UNITS:','ppm','ppm','ppm','ppb','ppb','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt','ppt']

# variable = ['YEARS','CO2EQ','KYOTO-CO2EQ','CO2','CH4','N2O','FGASSUMHFC134AEQ','MHALOSUMCFC12EQ','CF4','C2F6','C6F14','HFC23','HFC32','HFC43_10','HFC125','HFC134a','HFC143a','HFC227ea','HFC245fa','SF6','CFC_11','CFC_12','CFC_113','CFC_114','CFC_115','CARB_TET','MCF','HCFC_22','HCFC_141B','HCFC_142B','HALON1211','HALON1202','HALON1301','HALON2402','CH3BR','CH3CL']

# directory = '/Users/ph290/Documents/data/rcp_forcing_data/'

# files=['PRE2005_MIDYR_CONC.DAT','RCP3PD_MIDYR_CONC.DAT','RCP45_MIDYR_CONC.DAT','RCP6_MIDYR_CONC.DAT','RCP85_MIDYR_CONC.DAT']
# names=['historical','RCP2.5','RCP4.5','RCP6.0','RCP8.5']

# scenarios=[]

# for i in range(np.size(files)):
# 	data=np.genfromtxt(directory+files[i],skip_header=39)
# 	scenarios.append(data)

# '''
# plotting historical GHG concs
# '''

# vars_use=[5,3,4]

# ylim1=[260,260,600]
# ylim2=[330,400,1900]

# # fig = plt.figure()
# # for i in range(np.size(vars_use)):
# # 	ax = fig.add_subplot(3,1,i)
# # 	var_number=vars_use[i]
# #  	plt.xlim(1750,2010)
# # 	plt.ylim(ylim1[i],ylim2[i]) 	
# # 	a=scenarios[0][:,0]
# # 	b=scenarios[0][:,var_number]
# # 	a2=a[np.where(a >= 1960)]
# # 	b2=b[np.where(a >= 1960)]
# # 	ax.plot(a2,b2,linewidth=3)
# # 	plt.xlabel('year')	
# # 	plt.ylabel(variable[var_number]+' ('+units[var_number]+')')
# # 
# # plt.show()

# '''
# '''


# # fig = plt.figure()
# # for i in range(np.size(vars_use)):
# # 	ax = fig.add_subplot(3,1,i)
# # 	var_number=vars_use[i]
# #  	plt.xlim(1750,2010)
# # 	plt.ylim(ylim1[i],ylim2[i])
# # 	ax.plot(scenarios[0][:,0],scenarios[0][:,var_number],'r',linewidth=3)
# # 	plt.xlabel('year')	
# # 	plt.ylabel(variable[var_number]+' ('+units[var_number]+')')
# # 
# # plt.tight_layout()
# # plt.show()

# '''
# plot co2 concentrations for scenarios...
# '''

# plotting(3)
# # plotting2(3)

# '''
# mona loa
# '''

# # data=np.genfromtxt('/Users/ph290/Documents/data/co2_mm_mlo.txt',skip_header=72)
# # data[np.where(data[:,3] == -99.99),3]=np.nan
# # 
# # font_size=20
# # 
# # font = {'family' : 'normal',
# #         'weight' : 'bold',
# #         'size'   : font_size}
# # 
# # matplotlib.rc('font', **font)
# # 
# # lnwdth=4
# # 
# # ax = plt.plot(data[:,2],data[:,3],'red',linewidth=3)
# # plt.gca().spines['right'].set_linewidth(lnwdth) 
# # plt.gca().spines['left'].set_linewidth(lnwdth)
# # plt.gca().spines['bottom'].set_linewidth(lnwdth)
# # plt.gca().spines['top'].set_linewidth(lnwdth)
# # plt.title('Mauna Loa \'Keeling Curve\'')
# # plt.xlabel('year', fontsize=font_size, weight='bold')
# # plt.ylabel('atmospheric CO$_2$\nconcentration (ppm)', fontsize=font_size, weight='bold')
# # # plt.show()
# # # plt.tight_layout()
# # #plt.savefig('/Users/ph290/Documents/figures/mauna_loa.png', transparent=True)
