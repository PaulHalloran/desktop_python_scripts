import numpy as np
import numpy.ma as ma
import keyword

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

def carbchem(op_swtch,mdi,T,S,TCO2,TALK,Pr=0.0,TB=0.0,Ni=100.0,Tl=1.0e-5):
# This function calculates the inorganic carbon chemistry balance
# according to the method of Peng et al 1987
# The parameters are set in the first few lines

#salinity needs to be converted into psu *1000+35
#TCO2 and TALK must be in mol/kg /(1026.*1000.)
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

    msk1=ma.masked_greater_equal(T,mdi+1.0,copy=True)
    msk2=ma.masked_greater_equal(S,mdi+1.0,copy=True)
    msk3=ma.masked_greater_equal(TCO2,mdi+1.0,copy=True)
    msk4=ma.masked_greater_equal(TALK,mdi+1.0,copy=True)

    msk=np.ma.mask_or(msk1.mask,msk2.mask,copy=True)

    #create land-sea mask used by sea_msk.mask
    salmin = 1.0
    S2=np.copy(S)
    S2[np.abs(S) < salmin]=salmin

    tol = Tl
    mxiter = Ni

    op_fld = np.empty(T.shape)
    op_fld.fill(np.NAN)

    TB = np.ones(T.shape)
    TB[msk] = 4.106e-4*S2[msk]/35.0
    # this boron is from Peng

    #convert to Kelvin
    TK=np.copy(T[:])
    TK[msk] += +273.15

    alpha_s = np.ones(T.shape)
    alpha_s[msk] = np.exp( ( -60.2409 + 9345.17/TK[msk]  + 23.3585*np.log(TK[msk]/100.0) )  + ( 0.023517 - 0.023656*(TK[msk]/100.0) + 0.0047036*np.power((TK[msk]/100.0),2.0) )*S[msk] )
  
    K1 = np.ones(T.shape)
    K1[msk] = np.exp( ( -2307.1266/TK[msk] + 2.83655  - 1.5529413*np.log(TK[msk]) ) - ( 4.0484/TK[msk] + 0.20760841 )*np.sqrt(S[msk]) + 0.08468345*S[msk] - 0.00654208*np.power(S[msk],1.5) + np.log( 1.0 - 0.001005*S[msk] ) )

    if keyword.iskeyword(Pr):
        del_vol = np.ones(T.shape)
        del_com = np.ones(T.shape) 
        pf = np.ones(T.shape) 
        del_vol[msk] = -25.50 + 0.1271*T[msk]
        del_com[msk] = 1.0e-3*( -3.08 + 0.0877*T[msk] )
        pf[msk] = np.exp( ( 0.5*del_com[msk]*Pr[msk] - del_vol[msk] )*Pr[msk] / ( 83.131*TK[msk] ) )
        K1[msk] = K1[msk]*pf[msk]

    K2 = np.ones(T.shape)
    K2[msk] = np.exp( ( -3351.6106/TK[msk] - 9.226508 - 0.2005743*np.log(TK[msk]) ) - ( 23.9722/TK[msk] + 0.106901773 )*np.power(S[msk],0.5) + 0.1130822*S[msk] - 0.00846934*np.power(S[msk],1.5) + np.log( 1.0 - 0.001005*S[msk] ) )

    if keyword.iskeyword(Pr):
        del_vol = np.ones(T.shape)
        del_com = np.ones(T.shape) 
        pf = np.ones(T.shape) 
        del_vol[msk] = -15.82 - 0.0219*T[msk]
        del_com[msk] = 1.0e-3*( 1.13 - 0.1475*T[msk] )
        pf[msk] = np.exp( ( 0.5*del_com[msk]*Pr[msk] - del_vol[msk] )*Pr[msk] / ( 83.131*TK[msk] ) )
        K2[msk] = K2[msk]*pf[msk]

    KB = np.ones(T.shape)
    KB[msk] = np.exp( ( -8966.90 - 2890.53*np.power(S[msk],0.5) - 77.942*S[msk] + 1.728*np.power(S[msk],1.5)- 0.0996*np.power(S[msk],2.0) )/TK[msk] + ( 148.0248 + 137.1942*np.power(S[msk],0.5) + 1.62142*S[msk] ) - ( 24.4344 + 25.085*np.power(S[msk],0.5) + 0.2474*S[msk] )*np.log(TK[msk]) + 0.053105*(np.power(S[msk],0.5))*TK[msk] )

    if keyword.iskeyword(Pr):
        del_vol = np.ones(T.shape)
        del_com = np.ones(T.shape) 
        pf = np.ones(T.shape) 
        del_vol[msk] = -29.48 + 0.1622*T[msk]+ 0.0026080*np.power(T[msk],2.0)
        del_com[msk] = -2.84e-3
        pf[msk] = np.exp( ( 0.5*del_com[msk]*Pr[msk] - del_vol[msk] )*Pr[msk] / ( 83.131*TK[msk] ) )
        KB[msk] = KB[msk]*pf[msk]


    KW = np.ones(T.shape)
    KW[msk] = np.exp( ( -13847.26/TK[msk] + 148.96502 - 23.6521*np.log(TK[msk]) ) + ( 118.67/TK[msk] - 5.977 + 1.0495*np.log(TK[msk]) )*np.power(S[msk],0.5) - 0.01615*S[msk] )

    if keyword.iskeyword(Pr):
        del_vol = np.ones(T.shape)
        del_com = np.ones(T.shape) 
        pf = np.ones(T.shape) 
        del_vol[msk] = -25.60 + 0.2324*T[msk] - 0.0036246*np.power(T[msk],2.0)
        del_com[msk] = 1.0e-3*( -5.13 + 0.0794*T[msk] )
        pf[msk] = np.exp( ( 0.5*del_com[msk]*Pr[msk]- del_vol[msk] )*Pr[msk] / ( 83.131*TK[msk] ) )
        KW[msk] = KW[msk]*pf[msk]

    if ( op_swtch >= 6 or op_swtch <= 9 ):
        ca_conc = np.ones(T.shape)
        ca_conc[msk] = 0.01028*S2[msk]/35.0

    if ( op_swtch == 6 or op_swtch == 7 ):
        K_SP_C = np.ones(T.shape)
        K_SP_C[msk] = np.power(10.0,( ( -171.9065 - 0.077993*TK[msk] + 2839.319/TK[msk] + 71.595*np.log10(TK[msk]) ) + ( -0.77712 + 0.0028426*TK[msk] + 178.34/TK[msk] )*np.power(S[msk],0.5) - 0.07711*S[msk]+ 0.0041249*np.power(S[msk],1.5) ))
        if keyword.iskeyword(Pr):
            del_vol = np.ones(T.shape)
            del_com = np.ones(T.shape) 
            pf = np.ones(T.shape) 
            del_vol[msk] = -48.76 + 0.5304*T[msk]
            del_com[msk] = 1.0e-3*( -11.76 + 0.3692*T[msk] )
            pf[msk] = np.exp( ( 0.5*del_com[msk]*Pr[msk]   - del_vol[msk] )*Pr[msk] / ( 83.131*TK[msk] ) )
            K_SP_C[msk] = K_SP_C[msk]*pf[msk]

    if ( op_swtch == 8 or op_swtch == 9 ):
        K_SP_A = np.ones(T.shape)
        K_SP_A[msk] = np.power(10,( ( -171.945 - 0.077993*TK[msk] + 2903.293/TK[msk] + 71.595*np.log10(TK[msk]) ) + ( -0.068393 + 0.0017276*TK[msk] + 88.135/TK[msk] )*np.power(S[msk],0.5) - 0.10018*S[msk] + 0.0059415*np.power(S[msk],1.5) ))
        if keyword.iskeyword(Pr):
            del_vol = np.ones(T.shape)
            del_com = np.ones(T.shape) 
            pf = np.ones(T.shape) 
            del_vol[msk] = -46.0 + 0.5304*T[msk]
            del_com[msk] = 1.0e-3*( -11.76 + 0.3692*T[msk] )
            pf[msk] = np.exp( ( 0.5*del_com[msk]*Pr[msk]   - del_vol[msk] )*Pr[msk] / ( 83.131*TK[msk] ) )
            K_SP_A[msk] = K_SP_A[msk]*pf[msk]


    # Get first estimate for H+ concentration.
    aH = np.ones(T.shape)
    aH[msk] = 1.0e-8

    count = np.zeros(T.shape)
    tol_swtch = np.zeros(T.shape)

    AB = np.ones(T.shape)
    AC = np.ones(T.shape)
    AW = np.ones(T.shape)

    iter = 0
    test=2.0

    while  (test > 0.5 or iter >= mxiter):
      # Compute alkalinity guesses for Boron, Silicon, Phosphorus and Water
        AB[msk] = TB[msk]*KB[msk]/( aH[msk] + KB[msk] )

      #  ASi[msk] = TSi[msk]*KSi[msk]/( aH[msk] $
      #    + KSi[msk] )

      #  AP[msk] = TP[msk]*( 1.0/( 1.0 + KP2[msk]/aH[msk] $
      #    + KP2[msk]*KP3[msk]/(aH[msk]^2.0) ) + 2.0/( 1.0 $
      #    + aH[msk]/KP2[msk] + KP3[msk]/aH[msk] ) $
      #    + 3.0/( 1.0 + aH[msk]/KP3[msk] $
      #    + (aH[msk]^2.0)/(KP2[msk]*KP3[msk]) ) )

        AW[msk] = (KW[msk]/aH[msk]) - aH[msk]

      # using the guessed alkalinities and total alkalinity, calculate the
      # alkalinity due to carbon
      #  AC[msk] = TALK[msk] - ( AB[msk] + ASi[msk] $
      #    + AP[msk] + AW[msk] )
        AC[msk] = TALK[msk] - ( AB[msk] + AW[msk] )

      # and recalculate aH with the new As
        old_aH = np.copy(aH)
        aH[msk] = (0.5*K1[msk]/AC[msk])*( ( TCO2[msk] - AC[msk] ) + np.sqrt( ( TCO2[msk] - AC[msk] )*( TCO2[msk] - AC[msk] ) + 4.0*(AC[msk]*K2[msk]/K1[msk]) *( 2.0*TCO2[msk] - AC[msk] ) ) )

        tol_swtch[msk] = ( abs( ( aH[msk] - old_aH[msk] )/old_aH[msk] ) > tol )
        count[msk] = count[msk] + tol_swtch[msk]

        test = np.sum(tol_swtch)
        iter += 1

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

    denom[msk] = np.power(aH[msk],2.0) + K1[msk]*aH[msk] + K1[msk]*K2[msk]
    H2CO3[msk] = TCO2[msk]*np.power(aH[msk],2.0)/denom[msk]
    HCO3[msk] = TCO2[msk]*K1[msk]*aH[msk]/denom[msk]
    CO3[msk] = TCO2[msk]*K1[msk]*K2[msk]/denom[msk]

    pH[msk] = -np.log10(aH[msk])
    pCO2[msk] = H2CO3[msk]/alpha_s[msk]

    if ( op_swtch == 6 or op_swtch == 7 ):
        sat_CO3_C[msk] = K_SP_C[msk]/ca_conc[msk]
        if ( op_swtch == 7 ):
            sat_stat_C[msk] = CO3[msk]/sat_CO3_C[msk]

    if ( op_swtch == 8 or op_swtch == 9 ):
        sat_CO3_A[msk] = K_SP_A[msk]/ca_conc[msk]
        if ( op_swtch == 9 ):
            sat_stat_A[msk] = CO3[msk]/sat_CO3_A[msk]

    if ( op_swtch == 0 ):
        op_fld = np.zeros(T.shape)
        op_fld[msk] = count[msk]
    elif ( op_swtch == 1 ):
        op_fld[msk] = pCO2[msk]*1.0e6
    elif ( op_swtch == 2 ):
        op_fld[msk] = pH[msk]
    elif ( op_swtch == 3 ):
        op_fld[msk] = H2CO3[msk]
    elif ( op_swtch == 4 ):
        op_fld[msk] = HCO3[msk]
    elif ( op_swtch == 5 ):
        op_fld[msk] = CO3[msk]
    elif ( op_swtch == 6 ):
        op_fld[msk] = sat_CO3_C[msk]
    elif ( op_swtch == 7 ):
        op_fld[msk] = sat_stat_C[msk]
    elif ( op_swtch == 8 ):
        op_fld[msk] = sat_CO3_A[msk]
    elif ( op_swtch == 9 ):
        op_fld[msk] = sat_stat_A[msk]


    return op_fld

'''
test-data
'''

mdi=-999.0
sizing=(500,500)
T = np.empty(sizing)
S = np.empty(sizing)
TCO2 = np.empty(sizing)
TALK = np.empty(sizing)
T.fill(10.0)
S.fill(35.0)
TCO2.fill(0.0020)
TALK.fill(0.0022)
T[2,3]=mdi
S[3,3]=mdi
S[0,0]=0.5
TALK[2,3]=mdi
TCO2[2,3]=mdi

print carbchem(1,mdi,T,S,TCO2,TALK)

