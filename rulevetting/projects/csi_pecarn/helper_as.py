from os.path import join as oj

import numpy as np
import pandas as pd

'''Helper functions for dataset.py.
This file is optional.
'''

var_andy = ['AVPU', 'AgeInYears', 'AlteredMentalStatus', 'ArrPtIntub', 'Assault', 'AxialLoadAnyDoc',
           'CaseID', 'CervicalSpineImmobilization', 'ChildAbuse', 'Clotheslining', 
           'DxCspineInjury', 'Ethnicity', 'FallDownStairs', 'FallFromElevation',
            'FocalNeuroFindings', 'Gender', 'HeadFirst', 'HighriskDiving',
            'HighriskFall', 'HighriskHanging', 'HighriskHitByCar',
            'HighriskMVC', 'HighriskOtherMV', 'InjuryPrimaryMechanism',
           'IntervForCervicalStab', 'LOC', 'LimitedRangeMotion','LongTermRehab',
            'MVCDSC', 'MVCEFA', 'MVCHOC']

andy_highrisk = ['AlteredMentalStatus', 'LOC']


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
    
    as_binary1 = {
        'N': 0.,
        'Y': 1.,
        'ND': 0.,
    }  
    as_binary2 = {
            0:0.,
            1:1.,
    } 
    df.AVPU = df.AVPU.map(as_binary1)
    df.ArrPtIntub = df.ArrPtIntub.map(as_binary1)
    df.DxCspineInjury = df.DxCspineInjury.map(as_binary1)
    df.IntervForCervicalStab = df.IntervForCervicalStab.map(as_binary1)
    df.LongTermRehab = df.LongTermRehab.map(as_binary1)
    df.Clotheslining = df.Clotheslining.map(as_binary1)
    df.HeadFirst = df.HeadFirst.map(as_binary1)
    df.LimitedRangeMotion = df.LimitedRangeMotion.map(as_binary1)
    # FallDownStairs and FallFromElevation have weird coding(2, 3, etc)
    # MVC variables have weird coding with numbers that are not just (0, 1)
    
    
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



