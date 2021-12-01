from os.path import join as oj

import numpy as np
import pandas as pd

'''Helper functions for dataset.py.
This file is optional.
'''


def get_outcomes(RAW_DATA_PATH, NUM_PATIENTS=12044):
    """Read in the outcomes
    Returns
    -------
    outcomes: pd.DataFrame
        iai (has 761 positives)
        iai_intervention (has 203 positives)
    """
    form4abdangio = pd.read_csv(oj(RAW_DATA_PATH, 'form4bother_abdangio.csv')).rename(columns={'subjectid': 'id'})
    form6b = pd.read_csv(oj(RAW_DATA_PATH, 'form6b.csv')).rename(columns={'SubjectID': 'id'})
    form6c = pd.read_csv(oj(RAW_DATA_PATH, 'form6c.csv')).rename(columns={'subjectid': 'id'})

    # (6b) Intra-abdominal injury diagnosed in the ED/during hospitalization by any diagnostic method
    # 1 is yes, 761 have intra-abdominal injury
    # 2 is no -> remap to 0, 841 without intra-abdominal injury

    def get_ids(form, keys):
        '''Returns ids for which any of the keys is 1
        '''
        ids_all = set()
        for key in keys:
            ids = form.id.values[form[key] == 1]
            for i in ids:
                ids_all.add(i)
        return ids_all

    ids_iai = get_ids(form6b, ['IAIinED1'])  # form6b.id[form6b['IAIinED1'] == 1]

    # print(form4abdangio.keys())
    ids_allangio = get_ids(form4abdangio, ['AbdAngioVessel'])
    # print('num in 4angio', len(ids_allangio))
    # print(form6a.keys())
    # ids_alla = get_ids(form6a, ['DeathCause'])
    # print('num in a', len(ids_alla))
    # print(form6b.keys())
    ids_allb = get_ids(form6b, ['IVFluids', 'BldTransfusion'])
    # print('num in b', len(ids_allb))
    # print(form6c.keys())
    ids_allc = get_ids(form6c, ['IntervenDurLap'])
    # print('num in c', len(ids_allc))
    ids = ids_allb.union(ids_allangio).union(ids_allc)

    ids_iai_np = np.array(list(ids_iai)) - 1
    ids_np = np.array(list(ids)) - 1

    iai = np.zeros(NUM_PATIENTS).astype(int)
    iai[ids_iai_np] = 1
    iai_intervention = np.zeros(NUM_PATIENTS).astype(int)
    iai_intervention[ids_np] = 1

    df_iai = pd.DataFrame.from_dict({
        'id': np.arange(1, NUM_PATIENTS + 1),
        'iai': iai,
        'iai_intervention': iai_intervention
    })
    return df_iai


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
    
    comppain = {
        'N': 0.,
        'Y': 1.,
        'ND': 0.,
        'YND': 1.,
        'S': 1.,
        'P': 0.,
    }
    #     outcome = {
    #         "case": 1.,
    #         "ems": 0.,  
    #         "moi": 0.,  
    #         "ran": 0., 
    #     }    
    
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
    df.PtAmbulatoryPriorArrival=df.PtAmbulatoryPriorArrival.map(Y_binary)
    df.PtCompPain=df.PtCompPain.map(comppain)
    df.AVPU = df.AVPU.map(Y_binary)
    #df.ControlType = df.ControlType.map(outcome)
    df.ArrPtIntub = df.ArrPtIntub.map(Y_binary)
    df.DxCspineInjury = df.DxCspineInjury.map(Y_binary)
    df.IntervForCervicalStab = df.IntervForCervicalStab.map(Y_binary)
    df.LongTermRehab = df.LongTermRehab.map(Y_binary)
    #df.Clotheslining = df.Clotheslining.map(as_binary1)
    df.HeadFirst = df.HeadFirst.map(Y_binary)
    df.LimitedRangeMotion = df.LimitedRangeMotion.map(rangemotion)
    # FallDownStairs and FallFromElevation have weird coding(2, 3, etc)
    # MVC variables have weird coding with numbers that are not just (0, 1)
    
    df.CervicalSpineImmobilization=df.CervicalSpineImmobilization.map(csimmob)  
    
    ### SH ###
    #df.PtCompPainNeckMove = df.PtCompPainNeckMove.map(YN_binary)
    # small_freq_as_Y_binary = {
    #     "N": 0,
    #     "Y": 1,
    #     "ND": 1,
    #     "3": 1
    # }
    df.PtExtremityWeakness = df.PtExtremityWeakness.fillna("N").map(Y_binary)
    df.PtParesthesias = df.PtParesthesias.fillna("N").map(Y_binary)
    df.PtSensoryLoss = df.PtSensoryLoss.fillna("N").map(Y_binary)
    df.PtTender = df.PtTender.fillna("N").map(Y_binary)
    GCS_threshold = 15
    GCS_binary = {i:int(i<GCS_threshold) for i in range(0, 16)}
    GCS_binary[999] = 0 # SH : I set NaN as 0
    df.TotalGCS = df.TotalGCS.replace('7T', '7').fillna("999").astype(int).map(GCS_binary)
    #df.clotheslining = df.clotheslining.map(Y_binary)
    df.helmet = df.helmet.map(Y_binary)
    
   
    
    return df


def derived_feats(df):
    '''Add derived features
    '''
    binary = {
        0: 'no',
        1: 'yes',
        False: 'no',
        True: 'yes',
        'unknown': 'unknown'
    }
    df['AbdTrauma_or_SeatBeltSign'] = ((df.AbdTrauma == 'yes') | (df.SeatBeltSign == 'yes')).map(binary)
    df['AbdDistention_or_AbdomenPain'] = ((df.AbdDistention == 'AbdomenPain') | (df.SeatBeltSign == 'yes')).map(binary)
    df['Hypotension'] = (df['Age'] < 1 / 12) & (df['InitSysBPRange'] < 70) | \
                        (df['Age'] >= 1 / 12) & (df['Age'] < 5) & (df['InitSysBPRange'] < 80) | \
                        (df['Age'] >= 5) & (df['InitSysBPRange'] < 90)
    df['Hypotension'] = df['Hypotension'].map(binary)
    df['GCSScore_Full'] = (df['GCSScore'] == 15).map(binary)
    df['Age<2'] = (df['Age'] < 2).map(binary)
    df['CostalTender'] = ((df.LtCostalTender == 1) | (df.RtCostalTender == 1)).map(binary)  # | (df.DecrBreathSound)

    # Combine hispanic as part of race
    df['Race'] = df['Race_orig']
    df.loc[df.Hispanic == 'yes', 'Race'] = 'Hispanic'
    df.loc[df.Race == 'White', 'Race'] = 'White (Non-Hispanic)'
    df.drop(columns='Race_orig', inplace=True)

    return df

