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
    
    # add the suffix `_binary` to numeric variables with only 0 and 1 as non-nan inputs
    binary_numeric_cols = [col_name for col_name in numeric_data.columns\
                           if np.isin(numeric_data[col_name].dropna().unique(), [0, 1]).all()]
    numeric_data.columns = [col_name + '_binary' if col_name in binary_numeric_cols else col_name\
                                for col_name in numeric_data.columns]
    
    # convert text inputs to binary
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

def build_binary_covariates(df):
    '''
    Leonard et al. (2011) build robust versions of features derived solely on study site data.
    If a feature is an indicator of some condition at the study site, e.g. MedsGiven, then
    the robust version will be an indicator of the feature at the study site, outside hospital, or EMS.
    
    
    '''
    # Leonard et al. create robust analysis variables that end with `2` if the study site criterion was met
    # by EMS or outside data as well (three group OR). After discussing with Dr. Devlin, we believe a more
    # natural approach is to split by indicator at study site (already given) and indicator if condition improved
    # at study site re-evaluation We use boolean algebra to implement this.
    
    # first get the names of all robust analysis variables (other dfs have other covariates ending with 2)
    av_names = ['AlteredMentalStatus', 'LOC', 'NonAmbulatory', 'FocalNeuroFindings',
       'PainNeck', 'PosMidNeckTenderness', 'TenderNeck', 'Torticollis',
       'SubInj_Head', 'SubInj_Face', 'SubInj_Ext', 'SubInj_TorsoTrunk',
       'Predisposed', 'HighriskDiving', 'HighriskFall', 'HighriskHanging',
       'HighriskHitByCar', 'HighriskMVC', 'HighriskOtherMV', 'AxialLoadAnyDoc',
       'axialloadtop', 'Clotheslining']
    
    robust_av_names = [covar_name+'2_binary' for covar_name in av_names if covar_name+'2_binary' in df.columns.astype(str)]

    for robust_av in robust_av_names:
        base_av = robust_av[:-8] # strip off 2_binary
        df[base_av+'_improved'] = df[robust_av].copy()
        df[base_av+'_improved'][df[base_av+'_binary']==1] = 0 # condition remains indicated at study site
        # note we remove the `_binary` suffix, will do this for other variables later in this function
    df.drop(robust_av_names,axis=1,inplace=True)
  
    # for binary variables available before study site admission, create a similar `_improved` indicator
    all_covariates = df.columns
    ems_binary_var = pd.Series(all_covariates[all_covariates.str.endswith('_ems_binary')])
    site_covariates = all_covariates.difference(ems_binary_var)
    ems_binary_var = ems_binary_var.str.replace('_ems_binary', '', regex=True)
    
    outside_binary_var = pd.Series(all_covariates[all_covariates.str.endswith('_outside_binary')])
    site_covariates = site_covariates.difference(outside_binary_var)
    outside_binary_var = outside_binary_var.str.replace('_outside_binary', '', regex=True)

    binary_var = pd.Series(site_covariates[site_covariates.str.endswith('_binary')]).replace('_binary', '', regex=True)
    
    # get names of binary indicators with some form of outside check
    robust_var_names = set(binary_var).intersection(set(ems_binary_var).union(set(outside_binary_var))) 
    
    pd.options.mode.chained_assignment = None
    robust_var_names_removal = []
    for robust_var in robust_var_names:
        df[robust_var+'_improved_binary'] = 0
        
        if robust_var+'_ems_binary' in df.columns:
            df[robust_var+'_improved_binary'][(df[robust_var+'_binary'].copy()==0) &
                (df[robust_var+'_ems_binary'].copy()==1)] = 1 # condition no longer remains indicated at study site
            robust_var_names_removal.append(robust_var+'_ems_binary')
            
        if robust_var+'_outside_binary' in df.columns:
            df[robust_var+'_improved_binary'][(df[robust_var+'_binary'].copy()==0) &
                (df[robust_var+'_outside_binary'].copy()==1)] = 1  
            robust_var_names_removal.append(robust_var+'_outside_binary')
        
    df.drop(robust_var_names_removal,axis=1,inplace=True)
         
    # of data measured away from and at the study site, only GCS scores are not converted to improved

    df.columns = [col_name[:-7] if col_name.endswith('_binary') else col_name for col_name in df.columns]

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
    df.Race_posthoc = df.Race_posthoc.map(race)
    
    outcomeMap = {
        'PND': "Persistent Neurological Deficit",
        'N': "Normal",
        'DTH':"Death"
    }
    df.OutcomeStudySite_posthoc = df.OutcomeStudySite_posthoc.map(outcomeMap)

    neuroDeficit = {
        'NR': "Normal or good recovery",
        'MD': "Moderate disability",
        'SD':"Severe disability",
        'PVS':"Persistent vegetative state"
    }
    df.OutcomeStudySiteNeuro_posthoc = df.OutcomeStudySiteNeuro_posthoc.map(neuroDeficit)
    
    mobility = {
        'WD': "Wheelchair dependent",
        'I':"Immobile",
        'N':'Normal',
        'DA':'Dependent Ambulation'
    }
    df.OutcomeStudySiteMobility_posthoc=df.OutcomeStudySiteMobility_posthoc.map(mobility)

    
    return df


def derived_feats(df,veryyoung_age_cutoff=2,nonverbal_age_cutoff=5,young_adult_age_cutoff=12,stairs_cutoff=2):
    '''Add derived features
    '''
    df['VeryYoung'] = (df['AgeInYears'] < veryyoung_age_cutoff)
    df['NonVerbal'] = (df['AgeInYears'] < nonverbal_age_cutoff)
    df['YoungAdult'] = (df['AgeInYears'] >= young_adult_age_cutoff)
    df.drop(['AgeInYears'],axis=1,inplace=True)
        
    df['HighriskFallDownStairs'] = (df['FallDownStairs'].fillna(0) >= stairs_cutoff)    
    df.drop(['FallDownStairs'],axis=1,inplace=True)
    df.replace({False: 0., True: 1.}, inplace=True)

    # young children have difficulty localizing pain when asked
    # if a child is NonVerbal, this feature casts a wider net for neck pain complaints by including face and head
    # TODO: consider other regions
    pd.options.mode.chained_assignment = None
    df['PtCompPainNeck_robust']= df['PtCompPainNeck'].copy()
    
    df['PtCompPainNeck_robust'][(df['NonVerbal']==1.) & (df['VeryYoung'] == 0.) & 
                         ((df['PtCompPainNeck']==1.) | (df['PtCompPainHead']==1.) | (df['PtCompPainFace']==1.) |
                          (df['PtCompPainChest']==1.))
                         ] = 1
    
    # TODO: Make into a JC
    df.drop(['PtCompPainNeck'],axis=1,inplace=True)
    df = df.rename(columns={"PtCompPainNeck_robust": "PtCompPainNeck"})
        
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
    robust_an_names = [covar_name for covar_name in an_names if covar_name in df.columns]
    
    df.loc[:,'missing_rate'] = df[robust_an_names].isna().sum(axis = 1)/len(robust_an_names) # calculate missing fraction
    df = df[df.loc[:,'missing_rate'] < n] # drop observations with missing rate higer than n-fraction
    df.drop('missing_rate', axis=1, inplace=True)
    pd.options.mode.chained_assignment = 'warn'
    
    binary_covariates = [col_name for col_name in df.columns if ((len(pd.unique(df[col_name]))==2) |\
                                                                 (len(pd.unique(df[col_name]))==3))]
    binary_covariates.remove('OutcomeStudySite_posthoc') # boolean but encoded as string
            
    # fill binary NaN by "0"
    # Mean imputation removes most of the correlations in this data
    df[binary_covariates] = df[binary_covariates].fillna(0)
        
    return df