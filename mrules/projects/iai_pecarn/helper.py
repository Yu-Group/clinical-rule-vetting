from os.path import join as oj

import numpy as np
import pandas as pd
'''Helper functions for dataset.py
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
    race = {
        1: 'American Indian or Alaska Native',
        2: 'Asian',
        3: 'Black or African American',
        4: 'Native Hawaiian or other Pacific Islander',
        5: 'White',
        6: 'unknown',  # stated as unknown
        7: 'unknown'  # other
    }
    df.RACE = df.RACE.map(race)
    moi = {
        1: 'Motor vehicle collision',
        2: 'Fall from an elevation',
        3: 'Fall down stairs',
        4: 'Pedestrian/bicyclist struck by moving vehicle',
        5: 'Bike collision/fall',
        6: 'Motorcycle/ATV/Scooter collision',
        7: 'Object struck abdomen',
        8: 'unknown',  # unknown mechanism,
        9: 'unknown',  # other mechanism
        10: 'unknown'  # physician did not answer
    }
    df['MOI'] = df.RecodedMOI.map(moi)
    df = df.drop(columns=['RecodedMOI']).copy()
    abdTenderDegree = {
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'unknown',
        np.nan: 'unknown'
    }

    # combine aggregate gcs into total gcs
    idxs_to_replace = ~df['AggregateGCS'].isna() & df['GCSScore'].isna()
    df.loc[idxs_to_replace, 'GCSScore'] = df['AggregateGCS'][idxs_to_replace]

    # print(np.unique(df['AbdTenderDegree'], return_counts=True))
    df['AbdTenderDegree'] = df.AbdTenderDegree.map(abdTenderDegree)
    # print(np.unique(df['AbdTenderDegree'], return_counts=True))
    binary = {
        0: 'no',
        1: 'yes',
        False: 'no',
        True: 'yes',
        'unknown': 'unknown'
    }
    df['HISPANIC_ETHNICITY'] = (df['HISPANIC_ETHNICITY'] == '-1').map(
        binary)  # note: -1 is Hispanic (0 is not, 1 is unknown)

    # rename variables
    df = df.rename(columns={'RACE': 'Race_orig',
                            'SEX': 'Sex',
                            'HISPANIC_ETHNICITY': 'Hispanic',
                            'ageinyrs': 'Age'
                            })

    # set types of these variables to categorical
    ks_categorical = ['Sex', 'Race_orig', 'Hispanic',
                      'VomitWretch', 'MOI', 'ThoracicTender', 'ThoracicTrauma',
                      'DecrBreathSound', 'AbdDistention', 'AbdTenderDegree',
                      'AbdTrauma', 'SeatBeltSign', 'DistractingPain',
                      'AbdomenPain', 'AbdomenTender']
    for k in ks_categorical:
        df[k] = df[k].astype(str)

    df['AbdomenPain'] = df['AbdomenPain'].replace('3.0', 'other')
    df['CTScan'] = (df['AbdCTScan'] == 1.0).astype(int)

    # remap values which take on values 0....4
    ks_remap = ['VomitWretch',
                'ThoracicTender', 'ThoracicTrauma',
                'DecrBreathSound', 'AbdDistention',
                'AbdTrauma', 'SeatBeltSign',
                'DistractingPain', 'AbdomenPain', 'AbdomenTender']
    for k in ks_remap:
        vals = df[k].values
        is_na = df[k].isna()
        uniques = np.unique(vals).astype(str)
        contains_nan = np.sum(is_na) > 0
        if contains_nan and uniques.size in [4, 5] or ~contains_nan and uniques.size in [3, 4]:
            if '1.0' in uniques and '2.0' in uniques and ('3.0' in uniques or 'other' in uniques):
                df[k] = df[k].map({
                    '1.0': 'yes',
                    '2.0': 'no',
                    '3.0': 'unknown',
                    '4.0': 'unknown',
                    'other': 'other',
                    np.nan: 'unknown',
                })
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

    return df
