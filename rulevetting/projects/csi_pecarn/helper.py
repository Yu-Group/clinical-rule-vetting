from os.path import join as oj

import numpy as np
import pandas as pd
import re

'''Helper functions for dataset.py.
This file is optional.
'''

def assign_binary_outcome(s):
    '''
    Simple function for use in creating an outcome column in pandas df
    Returns 1 if a subject is a case (CSI injury)
    Else 0 (no CSI injury)
    '''
    if s == 'case':
        return 1
    return 0

def extract_numeric_data(input_df,categorical_covariates):
    '''
    This function returns a dataframe with all character columns dropped.
    Character variables which can be converted to binary such as 'Y'/'N' are mutated and kept
    Column names in categorical_covariates are unchachanged by this method
    '''
    categorical_data = input_df[categorical_covariates]
    noncat_data = input_df.drop(categorical_covariates,axis=1)

    numeric_data = noncat_data.select_dtypes([np.number]) # separate data that is already numeric
    char_data = noncat_data.select_dtypes([np.object]) # gets columns encoded as strings
    
    binary_data = pd.DataFrame(index=noncat_data.index) # init with study subject ID as index
    
    for column in char_data:
        if 'txt' in column: continue # don't process long-form text columns
        char_column = char_data[column] # select column
        unique_values = pd.unique(char_column) # get unique entries
        
        # encode yes as 1, no as 0
        if (('Y' in unique_values)|('A' in unique_values)|('YND' in unique_values)) & \
        (('N' in unique_values)|('ND' in unique_values)):
            conditions  = [char_column == 'Y',char_column == 'YD',char_column == 'YND',char_column == 'A',char_column == 'N']
            encodings = [1,1,1,1,0]
            binary_encoded = np.select(conditions, encodings, default=np.nan)
            col_name = column+"_binary"
            binary_data.loc[:,col_name] = binary_encoded.copy()
        # for clarity we convert the strings of post hoc outcomes into binary in the following loop
        else:
            conditions  = [char_column == 'Y', char_column == 'N',char_column == 'I',char_column == 'C',\
                           char_column == 'INTUB',char_column == 'EXTUB',char_column == 'NOTUB']
            encodings = [1,0,1,1,1,1,0]
            binary_encoded = np.select(conditions, encodings, default=np.nan)
            col_name = column+"_binary"
            binary_data.loc[:,col_name] = binary_encoded.copy()

    # add in newly created binary columns and removed categorical ones 
    output_df = pd.merge(numeric_data,binary_data,how="left",left_index=True,right_index=True)
    output_df = pd.merge(output_df,categorical_data,how="left",left_index=True,right_index=True)
    
    return output_df

def bin_continuous_data(input_df, binning_dict):
    '''
    This function bins and then one-hot encodes continuous covariates
    It returns a df with the cont. columns dropped and the new binary ones included
    
    Inputs
    ------
    input_df [pandas df]: raw data
    binning_dict [dictionary]: keys are column names; values are cutoffs to bin with
    '''
    # check that all columns are valid
    binning_cols = list(binning_dict.keys())
    bin_boolean = np.all(np.isin(binning_cols, input_df.columns))
    if not bin_boolean: raise ValueError("Invalid column name in `binning_dict`.") 
        
    for col_name, cutoff_tuple in binning_dict.items():
        
        # build appropriate names for bins
        cutoff_list = list(cutoff_tuple)
        cutoff_list.sort()
        if len(cutoff_list) <= 1: print("Cannot bin variables with single value")
        else:
            cutoff_names = [col_name+"_<"+str(cutoff_list[0])]
            range_names = [col_name+"_"+str(cutoff_list[i])+"-"+str(value) for i, value in enumerate(cutoff_list[1:])]
            cutoff_names.extend(range_names)
            cutoff_names.append(col_name+"_"+str(cutoff_list[-1])+"+")
            
        # create a temporary column of binned values
        bin_name = col_name+"_binned"
        cutoff_list.insert(0,-np.inf) # lower limit
        cutoff_list.append(np.inf) # upper limit
        input_df.loc[:,[bin_name]] = pd.cut(input_df.loc[:,col_name], cutoff_list, labels=cutoff_names)
        
        # convert bins to one-hot and drop working columms and original cont. one
        one_hot = pd.get_dummies(input_df.loc[:,bin_name])
        input_df = input_df.drop([col_name,bin_name],axis = 1)
        input_df = input_df.join(one_hot)

    return input_df

def build_robust_binary_covariates(df):
    '''
    Leonard et al. (2011) build robust versions of features derived solely on study site data.
    If a feature is an indicator of some condition at the study site, e.g. MedsGiven, then
    the robust version will be an indicator of the feature at the study site, outside hospital, or EMS.
    
    
    '''
    # TODO: functionalize
    # TODO: add comments
    
    covariates = df.columns
    ems_binary_var = pd.Series(covariates[covariates.str.endswith('_ems_binary')])
    site_covariates = covariates.difference(ems_binary_var)
    ems_binary_var = ems_binary_var.str.replace('_ems_binary', '', regex=True)
    
    outside_binary_var = pd.Series(covariates[covariates.str.endswith('_outside_binary')])
    site_covariates = site_covariates.difference(outside_binary_var)
    outside_binary_var = outside_binary_var.str.replace('_outside_binary', '', regex=True)

    binary_var = pd.Series(site_covariates[site_covariates.str.endswith('_binary')]).replace('_binary', '', regex=True)
    
    pd.options.mode.chained_assignment = None
    robust_var = set(binary_var).intersection(set(ems_binary_var).union(set(outside_binary_var)))
    
    for var in robust_var:
        robust_var = var + '_binary2'
        df[robust_var] = df[var+'_binary'].copy()
        if var + '_outside_binary' in covariates:
            df[robust_var] += df[var+'_outside_binary'].copy().fillna(0)
            df.drop(var+'_outside_binary',axis = 1, inplace=True)
        if var + '_ems_binary' in covariates:
            df[robust_var] += df[var+'_ems_binary'].copy().fillna(0)
            df.drop(var+'_ems_binary',axis = 1, inplace=True)
        df[robust_var][df[robust_var] >= 1] = 1.0
     
    covariates = df.columns
    ems_binary_var = pd.Series(covariates[covariates.str.endswith('_ems')])
    site_covariates = covariates.difference(ems_binary_var)
    ems_binary_var = ems_binary_var.str.replace('_ems', '', regex=True)
    
    outside_binary_var = pd.Series(covariates[covariates.str.endswith('_outside')])
    site_covariates = site_covariates.difference(outside_binary_var)
    outside_binary_var = outside_binary_var.str.replace('_outside', '', regex=True)
    
    robust_var = set(site_covariates).intersection(set(ems_binary_var).union(set(outside_binary_var)))
    for var in robust_var:
        if 'GCS' in var: continue
        robust_var = var + '2'
        df[robust_var] = df[var].copy()
        if var + '_outside' in covariates:
            df[robust_var] += df[var+'_outside'].copy().fillna(0)
            df.drop(var+'_outside',axis = 1, inplace=True)
        if var + '_ems' in covariates:
            df[robust_var] += df[var+'_ems'].copy().fillna(0)
            df.drop(var+'_ems',axis = 1, inplace=True)
        df[robust_var][df[robust_var] >= 1] = 1.0
    
    pd.options.mode.chained_assignment = 'warn'
    return df

def get_outcomes():
    """Read in the outcomes
    """
    # TODO: Implement?
    return


def rename_values(df):
    '''Map values to meanings
    Rename some features
    Compute a couple new features
    set types of
    '''

    # map categorical vars values
    race = {
        'AI': 'American Indian or Alaska Native',
        'A': 'Asian',
        'B': 'Black or African American',
        'PI': 'Native Hawaiian or other Pacific Islander',
        'W': 'White',
        'ND': 'unknown',  # stated as unknown
        'O': 'unknown'  # other
    }
    df.posthoc_Race = df.posthoc_Race.map(race)
    
    outcomeMap = {
        'PND': "Persistent Neurological Deficit",
        'N': "Normal",
        'DTH':"Death"
    }
    df.posthoc_OutcomeStudySite = df.posthoc_OutcomeStudySite.map(outcomeMap)

    neuroDeficit = {
        'NR': "Normal or good recovery",
        'MD': "Moderate disability",
        'SD':"Severe disability",
        'PVS':"Persistent vegetative state"
    }
    df.posthoc_OutcomeStudySiteNeuro = df.posthoc_OutcomeStudySiteNeuro.map(neuroDeficit)
    
    mobility = {
        'WD': "Wheelchair dependent",
        'I':"Immobile",
        'N':'Normal',
        'DA':'Dependent Ambulation'
    }
    df.posthoc_OutcomeStudySiteMobility=df.posthoc_OutcomeStudySiteMobility.map(mobility)

    
    return df


def derived_feats(df):
    '''Add derived features
    '''
    # TODO: Make JC on cutoffs  
    df['Age<2'] = (df['AgeInYears'] < 2)
    df['NonVerbal'] = (df['AgeInYears'] < 5)
    df['YoungAdult'] = (df['AgeInYears'] >= 12)
    df.drop(['AgeInYears'],axis=1,inplace=True)
        
    df['HighRiskFallDownStairs'] = (df['FallDownStairs'].fillna(0) >= 2)    
    df.drop(['FallDownStairs'],axis=1,inplace=True)
    df.replace({False: 0., True: 1.}, inplace=True)

    # young children have difficulty localizing pain when asked
    # if a child is NonVerbal, this feature casts a wider net for neck pain complaints by including face and head
    # TODO: consider other regions
    pd.options.mode.chained_assignment = None
    df['PainNeck_Robust']= df['PtCompPainNeck'].copy()
    
    df['PainNeck_Robust'][(df['NonVerbal']==1.) & (df['Age<2'] == 0.) & 
                         ((df['PtCompPainNeck']==1.) | (df['PtCompPainHead']==1.) | (df['PtCompPainFace']==1.))
                         ] = 1
    # TODO: Make into a JC
    df.drop(['PtCompPainNeck'],axis=1,inplace=True)
    
    df['PainNeck_Robust2']= df['PtCompPainNeck2'].copy()
    df['PainNeck_Robust2'][(df['NonVerbal']==1.) & (df['Age<2'] == 0.) & 
                         ((df['PtCompPainNeck2']==1.) | (df['PtCompPainHead2']==1.) | (df['PtCompPainFace2']==1.))
                         ] = 1
    df.drop(['PtCompPainNeck2'],axis=1,inplace=True)
    pd.options.mode.chained_assignment = 'warn'
    
    return df

def impute_missing_binary(df, n = 0.05):
    
    '''
    1. drop binary observations with missing rate higer than n% ;
    2. fill other NaN by "0";
    '''
    pd.options.mode.chained_assignment = None
    # drop observations
    an_names = ['AlteredMentalStatus', 'LOC', 'NonAmbulatory', 'FocalNeuroFindings',
       'PainNeck', 'PosMidNeckTenderness', 'TenderNeck', 'Torticollis',
       'SubInj_Head', 'SubInj_Face', 'SubInj_Ext', 'SubInj_TorsoTrunk',
       'Predisposed', 'HighriskDiving', 'HighriskFall', 'HighriskHanging',
       'HighriskHitByCar', 'HighriskMVC', 'HighriskOtherMV', 'AxialLoadAnyDoc',
       'axialloadtop', 'Clotheslining']
    robust_an_names = [covar_name if covar_name in df.columns else covar_name+'2' for covar_name in an_names]
    
    df.loc[:,'missing_rate'] = df[robust_an_names].isna().sum(axis = 1)/len(robust_an_names) # calculate missing fraction
    df = df[df.loc[:,'missing_rate'] < n] # drop observations with missing rate higer than n-fraction
    df.drop('missing_rate', axis=1, inplace=True)
    pd.options.mode.chained_assignment = 'warn'
    
    binary_covariates = [col_name for col_name in df.columns if ((len(pd.unique(df[col_name]))==2) |\
                                                                 (len(pd.unique(df[col_name]))==3))]
    binary_covariates.remove('posthoc_OutcomeStudySite') # boolean but encoded as string
            
    # fill binary NaN by "0"
    # Mean imputation removes most of the correlations in this data
    df[binary_covariates] = df[binary_covariates].fillna(0)
        
    return df