from os.path import join as oj

import numpy as np
import pandas as pd

'''Helper functions for dataset.py.
To do: 
    - any variable transformations or superfluous vars?
    - outcome vars: HospHeadPosCT, Intub24Head,
                        Neurosurgery, DeathTBI,
                        PosIntFinal, any else...
    - what to do about unknowns/NaNs
    - maybe rename columns to Python style
'''

def rename_tbi_neuro(df):
    """Rename categorical features in the TBI Neuro df
    Returns 
    -------
    df: pd.DataFrame - categorical vars are strings
    """
    # mapping binary variables to just yes and no
    binary0 = {
        0: 'No',
        1: 'Yes',
        np.nan: 'Unknown',
    }
    bool_cols0 = [col for col in df if np.isin(df[col].dropna().unique(), [0, 1]).all()]
    for bool_col in bool_cols0:
        df[bool_col] = df[bool_col].map(binary0)    
    
    # change type of categorical cols to strings
    categorical_cols = [col for col in df.columns.tolist() if col != 'id' and len(df[col].unique()) > 2]
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    
    return df

def rename_tbi_pud(df):
    """Rename categorical features in the TBI PUD df
    Returns
    -------
    df: pd.DataFrame - categorical vars are strings
    """
    empl_type = {
        1: 'Nurse Practitioner',
        2: 'Physician Assistant',
        3: 'Resident',
        4: 'Fellow',
        5: 'Faculty',
        # np.nan: 'Unknown'
    }
    df['EmplType'] = df['EmplType'].map(empl_type)

    cert_type = {
        1: 'Emergency Medicine',
        2: 'Pediatrics',
        3: 'Pediatrics Emergency Medicine',
        4: 'Emergency Medicine and Pediatrics',
        90: 'Other',
        # np.nan: 'Unknown'
    }
    df['Certification'] = df['Certification'].map(cert_type)

    inj_mech = {
        1: 'MVC',
        2: 'PedesMV',
        3: 'BikeMV',
        4: 'BikeCol',
        5: 'OtherWheelCrash',
        6: 'FallToGround',
        7: 'RanIntoStatObj',
        8: 'FallElev',  
        9: 'FallStair', 
        10: 'Sports',
        11: 'Assault',
        12: 'ObjStruckHead',
        90: 'Other',
        # np.nan: 'Unknown'
    }
    df['InjuryMech'] = df['InjuryMech'].map(inj_mech)

    inj_impact_sev = {
        1: 'Low',
        2: 'Moderate',
        3: 'High',
        # np.nan: 'Unknown'
    }
    df['High_impact_InjSev'] = df['High_impact_InjSev'].map(inj_impact_sev)

    # mapping binary variables to just yes and no
    # binary0 = {
    #     0: 'No',
    #     1: 'Yes',
    #     # np.nan: 'Unknown',
    # }
    # bool_cols0 = [col for col in df if np.isin(df[col].dropna().unique(), [0, 1]).all()]
    # for bool_col in bool_cols0:
    #     df[bool_col] = df[bool_col].map(binary0)

    # mapping binary variables to yes, no or unknown (for not applicable)
    # binary1 = {
        # 0: 'No',
        # 1: 'Yes',
        # 92: 'Not applicable',
        # np.nan: 'Unknown'
    # }    
    # bool_cols1 = [col for col in df if np.isin(df[col].dropna().unique(), [0, 1, 92]).all()]
    # for bool_col in bool_cols1:
        # df[bool_col] = df[bool_col].map(binary1)

    # verb = {
    #     0: 'No',
    #     1: 'Yes',
    #     91: 'Pre/Non-verbal',
    #     # np.nan: 'Unknown'
    # }  
    # bool_cols2 = [col for col in df if np.isin(df[col].dropna().unique(), [0, 1, 91]).all()]
    # for bool_col in bool_cols2:
    #     df[bool_col] = df[bool_col].map(verb)

    loc_separate = {
        0: 'No',
        1: 'Yes',
        2: 'Suspected',
        # np.nan: 'Unknown'
    }
    df['LOCSeparate'] = df['LOCSeparate'].map(loc_separate)

    loc_len = {
        1: '<5 sec',
        2: '5 sec - 1 min',
        3: '1-5 min',
        4: '>5 min',
        92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['LocLen'] = df['LocLen'].map(loc_len)
    
    seiz_occur = {
        1: 'Immediately on contact',
        2: 'Within 30 minutes of injury',
        3: '>30 minutes after injury',
        92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['SeizOccur'] = df['SeizOccur'].map(seiz_occur)
    
    seiz_len = {
        1: '<1 min',
        2: '1-5 min',
        3: '5-15 min',
        4: '>15 min',
        92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['SeizLen'] = df['SeizLen'].map(seiz_len)
    
    ha_severity = {
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['HASeverity'] = df['HASeverity'].map(ha_severity)
    
    ha_start = {
        1: 'Before head injury',
        2: 'Within 1 hr of event',
        3: '1-4 hrs after event',
        4: '>4 hrs after event',
        92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['HAStart'] = df['HAStart'].map(ha_start)
    
    vomit_nbr = {
        1: 'Once',
        2: 'Twice',
        3: '>2 times',
        92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['VomitNbr'] = df['VomitNbr'].map(vomit_nbr)
    
    vomit_start = {
        1: 'Before head injury',
        2: 'Within 1 hr of event',
        3: '1-4 hrs after event',
        4: '>4 hrs after event',
        92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['VomitStart'] = df['VomitStart'].map(vomit_start)
    
    vomit_last = {
        1: '<1 hr before ED',
        2: '1-4 hrs before ED',
        3: '>4 hrs before ED',
        # 92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['VomitLast'] = df['VomitLast'].map(vomit_last)
    
    gcs_eye = {
        1: 'None',
        2: 'Pain',
        3: 'Verbal',
        4: 'Spontaneous',
        # np.nan: 'Unknown'
    }
    df['GCSEye'] = df['GCSEye'].map(gcs_eye)
    
    gcs_verbal = {
        1: 'None',
        2: 'Incomprehensible sounds/moans',
        3: 'Inappropriate words/cries',
        4: 'Confused/cries',
        5: 'Oriented/coos',
        # np.nan: 'Unknown'
    }
    df['GCSVerbal'] = df['GCSVerbal'].map(gcs_verbal)
    
    gcs_motor = {
        1: 'None',
        2: 'Abnormal extension posturing',
        3: 'Abnormal flexing posturing',
        4: 'Pain withdraws',
        5: 'Localizes pain',
        6: 'Follow commands',
    }
    df['GCSMotor'] = df['GCSMotor'].map(gcs_motor)
    
    sfxpalp = {
        0: 'No',
        1: 'Yes',
        2: 'Unclear',
    }
    df['SFxPalp'] = df['SFxPalp'].map(sfxpalp)
        
    # map hermaloc to values
    hema_loc = {
        1: 'Frontal',
        2: 'Occipital',
        3: 'Parietal/Temporal',
        92: 'Not applicable',
    }
    df['HemaLoc'] = df['HemaLoc'].map(hema_loc)
    
    # mapping hermasize to values
    hema_size = {
        1: 'Small',
        2: 'Medium',
        3: 'Large',
        92: 'Not applicable',
        # np.nan: 'Unknown'
    }
    df['HemaSize'] = df['HemaSize'].map(hema_size)
    
    # mapping gender to value
    gender = {
        1: 'Male',
        2: 'Female',
        # np.nan: 'Unknown'
    }
    df['Gender'] = df['Gender'].map(gender)
    
    
    # mapping ethnicity to names
    eth = {
        1: 'Hispanic',
        2: 'Non-Hispanic',
        # np.nan: 'Unknown'
    }
    df['Ethnicity'] = df['Ethnicity'].map(eth)
    
    # mapping race to names
    races = {
        1: 'White',
        2: 'Black',
        3: 'Asian',
        4: 'American Indian',
        5: 'Pacific Islander',
        90: 'Other',
        # np.nan: 'Unknown'
    }
    df['Race'] = df['Race'].map(races)
    
    # mapping ed disposition to names
    ed_disposition = {
        1: 'Home',
        2: 'OR',
        3: 'Admit - general patient',
        4: 'Short-stay\Observation',
        5: 'ICU',
        6: 'Transferred Hospital',
        7: 'AMA',
        8: 'Death in ED',
        90: 'Other',
        # np.nan: 'Unknown'
    }
    df['EDDisposition'] = df['EDDisposition'].map(ed_disposition)

    # make all of these columns categorical
    # numeric_cols = ['id', 'GCSTotal', 'AgeInMonths', 'AgeinYears']
    # categorical_cols = [col for col in df.columns.tolist() if col not in numeric_cols and len(df[col].unique()) > 2]
    # for col in categorical_cols:
    #     df[col] = df[col].astype(str)

    return df
