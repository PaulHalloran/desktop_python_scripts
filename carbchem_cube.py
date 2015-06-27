'''
Input should be:
Temperature in K
S in PSU (I think)
DIC and ALK in MOL? values about 2.0
i.e. CMIP5 output
'''

import numpy as np
import numpy.ma as ma
import keyword
import warnings
import iris.quickplot as qplt
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

print'  ops= 0 ;  output is iteration count'
print'       1 ;            pCO2'
print'       2 ;            pH'
print'       3 ;            [H2CO3]'
print'       4 ;            [HCO3]'
print'       5 ;            [CO3]'
print'       6 ;            satn [co3] : calcite'
print'       7 ;            saturation state: calcite'
print'       8 ;            satn [CO3] : aragonite'
print'       9 ;            saturation state: aragonite'

print 'inputs: op_swtch,mdi,T,S,TCO2,TALK'


def pressure_fun(a,b,c,d,e,T):
    del_vol = np.ones(T.shape, dtype='f')
    del_com = np.ones(T.shape, dtype='f') 
    pf = np.ones(T.shape, dtype='f')
    del_vol = a + b *T + c * np.power(T,2.0)
    del_com = 1.0e-3*( d + e*T )
    pf = np.exp( ( 0.5*del_com*Pr   - del_vol )*Pr / ( 83.131*TK ) )
    return pf

def carbiter(T, TCO2, TALK, TB, msk, tol, mxiter, K1, K2, KB, KW):
    aH = np.empty_like(T, dtype='f')
    aH.fill(1.0e-8)
    count = np.zeros_like(T)
    tol_swtch = np.zeros_like(T)
#MB -
#    AB = np.ones(T.shape)
#    AC = np.ones(T.shape)
#    AW = np.ones(T.shape)
    
    #MB+
    TBKB = TB * KB
    K2_K1x4 = 4.0 * K2 / K1
    K2_2 = 0.5 * K1
    #
    
    iter = 0
    test = 2.0
    while test > 0.5 and iter < mxiter:
        # Compute alkalinity guesses for Boron, Silicon, Phosphorus and Water
        #MB- AB = TB * KB / (aH + KB)
        #AB = TBKB / (aH + KB)
        AB = np.divide(TBKB,(aH + KB))
        #  ASi = TSi*KSi/( aH $
        #    + KSi )
        #  AP = TP*( 1.0/( 1.0 + KP2/aH $
        #    + KP2*KP3/(aH^2.0) ) + 2.0/( 1.0 $
        #    + aH/KP2 + KP3/aH ) $
        #    + 3.0/( 1.0 + aH/KP3 $
        #    + (aH^2.0)/(KP2*KP3) ) )
        AW = (KW / aH) - aH
        # using the guessed alkalinities and total alkalinity, calculate the
        # alkalinity due to carbon
        #  AC = TALK - ( AB + ASi $
        #    + AP + AW )
        AC = TALK - (AB + AW)
        # and recalculate aH with the new As
        #MB+
        TCO2_AC = TCO2 - AC
        #
        old_aH = np.copy(aH)
        #MB- aH = (0.5 * K1 / AC) * ((TCO2 - AC) + np.sqrt((TCO2 - AC) * (TCO2 - AC) + 4.0 * (AC * K2 / K1) * (2.0 * TCO2 - AC)))
        temp = TCO2_AC*TCO2_AC + (AC * K2_K1x4) * (2.0 * TCO2 - AC)
        aH = (K2_2 / AC) * (TCO2_AC + np.sqrt(temp))
        tol_swtch = abs((aH - old_aH) / old_aH) > tol
        count = count + tol_swtch
        test = np.sum(tol_swtch)
        iter += 1
        
    #aH[~msk] = 1.0
    #count[~msk] = 0
    return aH, count

def carbchem(op_swtch,mdi,T_cube,S_cube,TCO2_cube,TALK_cube,Pr=0.0,TB=0.0,Ni=100.0,Tl=1.0e-5):
# This function calculates the inorganic carbon chemistry balance
# according to the method of Peng et al 1987
# The parameters are set in the first few lines

#salinity needs to be converted into psu
#TCO2 and TALK must be in mol/kg
#the ones below here are not needed

# This procedure calculates the inorganic carbon chemistry balance
# according to the method of Peng et al 1987
# The parameters are set in the first few lines
#
#  ops= 0 ;  output is iteration count
#       1 ;            pCO2
#       2 ;            pH
#       3 ;            [H2CO3]
#       4 ;            [HCO3]
#       5 ;            [CO3]
#       6 ;            satn [co3] : calcite
#       7 ;            saturation state: calcite
#       8 ;            satn [CO3] : aragonite
#       9 ;            saturation state: aragonite

    #make sure grids are same size
    #make sure rthey years are the same
    #extarct the data from the cubes
    
# from iris import *
# from iris.analysis import *
# import iris.analysis
# from numpy import *
# from matplotlib.pyplot import *
# from scipy.stats.mstats import *
# import iris.plot as iplt
# import seawater
# import numpy
# import iris.quickplot as quickplot
# import iris.analysis.stats as istats
# temp = iris.load_cube('/home/ph290/tmp/hadgem2es_potential_temperature_historical_regridded.nc').extract(Constraint(depth = 0))
# sal = iris.load_cube('/home/ph290/tmp/hadgem2es_salinity_historical_regridded.nc').extract(Constraint(depth = 0))
# carb = iris.load_cube('/home/ph290/tmp/hadgem2es_dissolved_inorganic_carbon_historical_regridded.nc').extract(Constraint(depth = 0))
# alk = iris.load_cube('/home/ph290/tmp/hadgem2es_total_alkalinity_historical_regridded.nc').extract(Constraint(depth = 0))
# import carbchem
# co2 = carbchem.carbchem(1,temp.data.fill_value,temp,sal,carb,alk)
# T_cube = temp
# S_cube = sal
# TCO2_cube = carb
# TALK_cube = alk  
# mdi = temp.data.fill_value
	
    t_lat = np.size(T_cube.coord('latitude').points)    
    s_lat = np.size(S_cube.coord('latitude').points)
    c_lat = np.size(TCO2_cube.coord('latitude').points)
    a_lat = np.size(TALK_cube.coord('latitude').points)
    lat_test = t_lat == s_lat == c_lat == a_lat

    t_lon = np.size(T_cube.coord('longitude').points) 
    s_lon = np.size(S_cube.coord('longitude').points)
    c_lon = np.size(TCO2_cube.coord('longitude').points)
    a_lon = np.size(TALK_cube.coord('longitude').points)
    lon_test = t_lon == s_lon == c_lon == a_lon

    if lat_test and lon_test:
	coord = T_cube.coord('time')
	T_year = np.array([coord.units.num2date(value).year for value in coord.points])
	coord = S_cube.coord('time')
	S_year = np.array([coord.units.num2date(value).year for value in coord.points])
	coord = TCO2_cube.coord('time')
	TCO2_year = np.array([coord.units.num2date(value).year for value in coord.points])
	coord = TALK_cube.coord('time')
	TALK_year = np.array([coord.units.num2date(value).year for value in coord.points])

	common_yrs = np.intersect1d(T_year,S_year)
	common_yrs = np.intersect1d(common_yrs,TCO2_year)
	common_yrs = np.intersect1d(common_yrs,TALK_year)

	if not (common_yrs.size == T_year.size | common_yrs.size == S_year.size | common_yrs.size == TCO2_year.size | common_yrs.size == TALK_year.size):
		print 'warning: timeseries shortened so all variables have the same years'
		T_cube = T_cube[np.nonzero(np.in1d(T_year,common_yrs))[0]]
		S_cube = S_cube[np.nonzero(np.in1d(S_year,common_yrs))[0]]
		TCO2_cube = TCO2_cube[np.nonzero(np.in1d(TCO2_year,common_yrs))[0]]
		TALK_cube = TALK_cube[np.nonzero(np.in1d(TALK_year,common_yrs))[0]]

	output_cube = T_cube.copy()
	T_cube = T_cube-273.15
	T = T_cube.data.copy()
	S = S_cube.data.copy()
	TCO2_cube = TCO2_cube/1026.0
	TCO2 = TCO2_cube.data.copy()
	TALK_cube = TALK_cube/1026.0
	TALK = TALK_cube.data.copy()

	msk1=ma.masked_greater_equal(T,mdi-1.0,copy=True)
	msk2=ma.masked_greater_equal(S,mdi-1.0,copy=True)
	msk3=ma.masked_greater_equal(TCO2,mdi-1.0,copy=True)
	msk4=ma.masked_greater_equal(TALK,mdi-1.0,copy=True)

	msk=msk1.mask | msk2.mask | msk3.mask | msk4.mask

	T[msk]=np.nan
	S[msk]=np.nan
	TALK[msk]=np.nan
	TCO2[msk]=np.nan

	# T = np.array([13.74232016,25.0])
	# S = np.array([33.74096661,35.0])
	# TCO2 = np.array([0.0019863,2.0e-3])
	# TALK = np.array([0.00226763,2.2e-3])
	# msk = ma.masked_greater_equal(T,mdi-1.0,copy=True)

	#create land-sea mask used by sea_msk.mask
	salmin = 1.0
	S2=np.copy(S)
	S2[np.abs(S) < salmin]=salmin

	tol = Tl
	mxiter = Ni

	op_fld = np.empty(T.shape)
	op_fld.fill(np.NAN)

	#    TB = np.ones(T.shape)
	#    TB = 4.106e-4*S2/35.0
	TB = np.empty_like(T)
	TB = np.multiply(S2,4.106e-4/35.0, TB)
	# this boron is from Peng

	#convert to Kelvin
	TK=np.copy(T[:])
	TK += +273.15

	alpha_s = np.ones(T.shape)
	alpha_s = np.exp( ( -60.2409 + 9345.17/TK  + 23.3585*np.log(TK/100.0) )  + ( 0.023517 - 0.023656*(TK/100.0) + 0.0047036*np.power((TK/100.0),2.0) )*S )

	K1 = np.ones(T.shape)
	K1 = np.exp( ( -2307.1266/TK + 2.83655  - 1.5529413*np.log(TK) ) - ( 4.0484/TK + 0.20760841 )*np.sqrt(S) + 0.08468345*S - 0.00654208*np.power(S,1.5) + np.log( 1.0 - 0.001005*S ) )

	a = np.array([-25.50,-15.82,-29.48,-25.60,-48.76,-46.0])
	b = np.array([0.1271,0.0219,0.2324,0.5304,0.5304])
	c = np.array([0.0,0.0,0.0026080,0.0036246,0.0,0.0])
	d = np.array([-3.08,1.13,(-2.84e-3)/(1.0e-3),-5.13,-11.76,-11.76])
	e = np.array([0.0877,0.1475,0.0,0.0794,0.3692,0.3692])

	if keyword.iskeyword(Pr):
	    instance = 0
	    pf = pressure_fun(a[instance],b[instance],c[instance],d[instance],e[instance],T)
	    K1 = K1*pf

	K2 = np.ones(T.shape)
	K2 = np.exp( ( -3351.6106/TK - 9.226508 - 0.2005743*np.log(TK) ) - ( 23.9722/TK + 0.106901773 )*np.power(S,0.5) + 0.1130822*S - 0.00846934*np.power(S,1.5) + np.log( 1.0 - 0.001005*S ) )

	if keyword.iskeyword(Pr):
	    instance = 1
	    pf = pressure_fun(a[instance],b[instance],c[instance],d[instance],e[instance],T)
	    K2 = K2*pf

	KB = np.ones(T.shape)
	KB = np.exp( ( -8966.90 - 2890.53*np.power(S,0.5) - 77.942*S + 1.728*np.power(S,1.5)- 0.0996*np.power(S,2.0) )/TK + ( 148.0248 + 137.1942*np.power(S,0.5) + 1.62142*S ) - ( 24.4344 + 25.085*np.power(S,0.5) + 0.2474*S )*np.log(TK) + 0.053105*(np.power(S,0.5))*TK )

	if keyword.iskeyword(Pr):
	    instance = 2
	    pf = pressure_fun(a[instance],b[instance],c[instance],d[instance],e[instance],T)
	    KB = KB*pf

	KW = np.ones(T.shape)
	KW = np.exp( ( -13847.26/TK + 148.96502 - 23.6521*np.log(TK) ) + ( 118.67/TK - 5.977 + 1.0495*np.log(TK) )*np.power(S,0.5) - 0.01615*S )

	if keyword.iskeyword(Pr):
	    instance = 3
	    pf = pressure_fun(a[instance],b[instance],c[instance],d[instance],e[instance],T)
	    KW = KW*pf

	if ( op_swtch >= 6 or op_swtch <= 9 ):
	    ca_conc = np.ones(T.shape)
	    ca_conc = 0.01028*S2/35.0

	if ( op_swtch == 6 or op_swtch == 7 ):
	    K_SP_C = np.ones(T.shape)
	    K_SP_C = np.power(10.0,( ( -171.9065 - 0.077993*TK + 2839.319/TK + 71.595*np.log10(TK) ) + ( -0.77712 + 0.0028426*TK + 178.34/TK )*np.power(S,0.5) - 0.07711*S+ 0.0041249*np.power(S,1.5) ))
	    if keyword.iskeyword(Pr):
		    instance = 4
		    pf = pressure_fun(a[instance],b[instance],c[instance],d[instance],e[instance],T)
		    K_SP_C = K_SP_C*pf


	if ( op_swtch == 8 or op_swtch == 9 ):
	    K_SP_A = np.ones(T.shape)
	    K_SP_A = np.power(10,( ( -171.945 - 0.077993*TK + 2903.293/TK + 71.595*np.log10(TK) ) + ( -0.068393 + 0.0017276*TK + 88.135/TK )*np.power(S,0.5) - 0.10018*S + 0.0059415*np.power(S,1.5) ))
	    if keyword.iskeyword(Pr):
		    instance = 5
		    pf = pressure_fun(a[instance],b[instance],c[instance],d[instance],e[instance],T)
		    K_SP_A = K_SP_A*pf


	# Get first estimate for H+ concentration.

	aH, count = carbiter(T, TCO2, TALK, TB, msk, tol, mxiter, K1, K2, KB, KW)

	# now we have aH we can calculate...
	denom = np.zeros(T.shape)
	H2CO3 = np.zeros(T.shape)
	HCO3 = np.zeros(T.shape)
	CO3 = np.zeros(T.shape)
	pH = np.zeros(T.shape)
	pCO2 = np.zeros(T.shape)
	if ( op_swtch == 6 or op_swtch == 7 ):
	    sat_CO3_C = np.zeros(T.shape)
	if ( op_swtch == 7 ):
	    sat_stat_C = np.zeros(T.shape)
	if ( op_swtch == 8 or op_swtch == 9 ):
	    sat_CO3_A = np.zeros(T.shape)
	if ( op_swtch == 9 ):
	    sat_stat_A = np.zeros(T.shape)

	denom = np.power(aH,2.0) + K1*aH + K1*K2
	H2CO3 = TCO2*np.power(aH,2.0)/denom
	HCO3 = TCO2*K1*aH/denom
	CO3 = TCO2*K1*K2/denom

	pH = -np.log10(aH)
	pCO2 = H2CO3/alpha_s

	if ( op_swtch == 6 or op_swtch == 7 ):
	    sat_CO3_C = K_SP_C/ca_conc
	    if ( op_swtch == 7 ):
		    sat_stat_C = CO3/sat_CO3_C

	if ( op_swtch == 8 or op_swtch == 9 ):
	    sat_CO3_A = K_SP_A/ca_conc
	    if ( op_swtch == 9 ):
		    sat_stat_A = CO3/sat_CO3_A

	output_cube = output_cube*0.0+np.nan
	if ( op_swtch == 0 ):
	    op_fld = np.zeros(T.shape)
	    op_fld = count
	elif ( op_swtch == 1 ):
	    output_cube.data = pCO2*1.0e6
	    output_cube.standard_name = 'surface_partial_pressure_of_carbon_dioxide_in_sea_water'
	    output_cube.long_name = 'CO2 concentration'
	    output_cube.units = 'uatm'
	elif ( op_swtch == 2 ):
	    output_cube.data = pH
	    output_cube.standard_name = 'sea_water_ph_reported_on_total_scale'
	    output_cube.long_name = 'pH'
	    output_cube.units = '1'
	elif ( op_swtch == 3 ):
	    output_cube.data = H2CO3
	elif ( op_swtch == 4 ):
	    output_cube.data = HCO3
	elif ( op_swtch == 5 ):
	    output_cube.data = CO3
	elif ( op_swtch == 6 ):
	    output_cube.data = sat_CO3_C
	elif ( op_swtch == 7 ):
	    output_cube.data = sat_stat_C
	elif ( op_swtch == 8 ):
	    output_cube.data = sat_CO3_A
	elif ( op_swtch == 9 ):
	    output_cube.data = sat_stat_A

	return output_cube
 


