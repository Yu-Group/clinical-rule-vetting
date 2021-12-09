from os.path import join as oj

import numpy as np
import pandas as pd

'''Helper functions for dataset.py.
This file is optional.
'''

# def get_outcomes(RAW_DATA_PATH, NUM_PATIENTS=12044):
#     return NotImplemented

def rename_values(df):
    '''Map values to meanings
    Rename some features
    Compute a couple new features
    set types of
    '''

    # map categorical vars values
    
    Y_binary ={ 
        'Y': 1.,
        'N': 0.,
        'ND': 0.,
        'YND': 0.,
        '3': 0.,
        'S': 0.,
        'P': 0.
    }
    
    zeroone_binary = {
            0:0.,
            1:1.,
    } 
    
    AVPU = {
        'A':0.,
        'V':1.,
        'P':1.,
        'U':1.,
        'N':0.,
        0.:0.,
    }
    
    comppain = {
        'N': 0.,
        'Y': 1.,
        'ND': 0.,
        'YND': 1.,
        'S': 1.,
        'P': 1.,
    }
     
    rangemotion = {
        'N': 0.,
        'Y': 1.,
        'ND': 0.,
        '3': 1.,
        '4': 1.,
    }
    
    motorgcs={
        1.:0.,
        2.:0.,
        3.:0.,
        4.:0.,
        5.:0.,
        6.:1.
    }
    csimmob={3.: 0., 
             1.: 1.,
             2.: 1.}
    
    df.MedsGiven=df.MedsGiven.map(Y_binary)
    df.MedsRecdPriorArrival=df.MedsRecdPriorArrival.map(Y_binary)
    df.Predisposed=df.Predisposed.map(zeroone_binary)
    df.MotorGCS=df.MotorGCS.map(motorgcs)
    
    df.PtCompPain=df.PtCompPain.map(comppain)
    df.AVPUDetails = df.AVPUDetails.fillna(0.)
    df.AVPUDetails = df.AVPUDetails.map(AVPU)
    df.ArrPtIntub = df.ArrPtIntub.map(Y_binary)
    df.DxCspineInjury = df.DxCspineInjury.map(Y_binary)
    df.IntervForCervicalStab = df.IntervForCervicalStab.map(Y_binary)
    df.LongTermRehab = df.LongTermRehab.map(Y_binary)
    
    df.LimitedRangeMotion = df.LimitedRangeMotion.map(rangemotion)
    df.CervicalSpineImmobilization=df.CervicalSpineImmobilization.map(csimmob)  
    df.PtExtremityWeakness = df.PtExtremityWeakness.fillna("N").map(Y_binary)
    df.PtParesthesias = df.PtParesthesias.fillna("N").map(Y_binary)
    df.PtSensoryLoss = df.PtSensoryLoss.fillna("N").map(Y_binary)
    df.PtTender = df.PtTender.fillna("N").map(Y_binary)
    df.helmet = df.helmet.map(Y_binary)
    

    #GCS_threshold = 15
    #GCS_binary = {i:int(i<GCS_threshold) for i in range(0, 16)}
    #GCS_binary[999] = 0 # SH : I set NaN as 0
    #df.TotalGCS = df.TotalGCS.replace('7T', '7').fillna("999").astype(int).map(GCS_binary)
    
    #df.ControlType = df.ControlType.map(outcome)
    #df.Clotheslining = df.Clotheslining.map(as_binary1)
    # FallDownStairs and FallFromElevation have weird coding(2, 3, etc)
    # MVC variables have weird coding with numbers that are not just (0, 1)
    #df.HeadFirst = df.HeadFirst.map(Y_binary)
    #df.PtAmbulatoryPriorArrival=df.PtAmbulatoryPriorArrival.map(Y_binary)
    #df.PtCompPainNeckMove = df.PtCompPainNeckMove.map(YN_binary)
    #df.clotheslining = df.clotheslining.map(Y_binary)
    
    return df

# def derived_feats(df):
#     return NotImplemented

