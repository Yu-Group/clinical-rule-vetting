from os.path import join as oj

import numpy as np
import os
import random
import pandas as pd
import re
from tqdm import tqdm
from typing import Dict
from joblib import Memory

import rulevetting
import rulevetting.api.util
from rulevetting.projects.csi_pecarn import helper
from rulevetting.templates.dataset import DatasetTemplate


class Dataset(DatasetTemplate):
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        # all the fnames to be loaded and searched over        
        fnames = sorted([fname for fname in os.listdir(raw_data_path) if 'csv' in fname])
        
        # suffix to covariates for differentiation between source datasets
        suffix_dictionary = {'analysisvariables.csv':'',
                             'clinicalpresentationsite.csv':'',
                             'clinicalpresentationoutside.csv':'_outside',
                            'clinicalpresentationfield.csv':'_ems',
                            'demographics.csv':'_posthoc',
                            'injuryclassification.csv':'_posthoc',
                            'injurymechanism.csv':'',
                            'kappa.csv':'',
                            'medicalhistory.csv':'',
                            'radiologyoutside.csv':'_outside',
                            'radiologyreview.csv':'_posthoc',
                            'radiologysite.csv':'_posthoc'}
            
        # read through each fname and save into the r dictionary
        r = {}
        print('read all the csvs...\n', fnames)
        if len(fnames) == 0:
            print('no csvs found in path', raw_data_path)
        
        # replace studysubjectid cases with id
        for fname in tqdm(fnames):
            df = pd.read_csv(oj(raw_data_path, fname), encoding="ISO-8859-1")
            df.columns = [re.sub('StudySubjectID','id',x,flags=re.IGNORECASE) for x in df.columns]
            df.columns = [re.sub('SITE','site',x) for x in df.columns]
            df.columns = [re.sub('CaseID','case_id',x,flags=re.IGNORECASE) for x in df.columns]
            df.columns = [re.sub('ControlType','control_type',x,flags=re.IGNORECASE) for x in df.columns]
            df.columns = [re.sub('^CSpine','CervicalSpine',x) for x in df.columns]
            df.columns = [re.sub('^CS','CervicalSpine',x) for x in df.columns]
            df.columns = [re.sub('^subinj_','SubInj_',x) for x in df.columns]
                            
            assert ('id' in df.keys())
            df = df.set_index(['id','case_id','site','control_type']) # use a multiIndex
            
            # add suffix to distinguish original datasets
            covar_suffix = suffix_dictionary[fname]
            df.columns = df.columns.astype(str) + covar_suffix

            r[fname] = df

        # Get filenames we consider in our covariate analysis
        # We do not consider radiology data or injury classification because this data is not
        # available at decision time in the ED.
        fnames_small = [fname for fname in fnames
                        if not 'radiologyreview' in fname
                        and not 'analysisvariables' in fname
                        and not 'kappa' in fname]
        
        df = r['analysisvariables.csv']
                
        print('merging all of the dfs...')
        for i, fname in tqdm(enumerate(fnames_small)):
            df2 = r[fname].copy()
            df = pd.merge(df,df2,how="left",left_index=True,right_index=True)
        
        
        # judgement call to use kappa variables where appropriate
        if kwargs['use_kappa']:
            kappa_data = r['kappa.csv']
            
            kappa_rename_dict = {
                'EDDocumentation':'EDDocumentation_outside',
                'FieldDocumentation':'FieldDocumentation_ems',
                'PatientsPosition':'PatientsPosition_ems',
                'PtAmbulatoryPriorEMSArrival':'PtAmbulatoryPriorEMSArrival_ems',
                'ShakenBabySyndrome':'ShakenBabySyndrome_posthoc', 'clotheslining':'Clotheslining'
            }

            kappa_data.rename(columns=kappa_rename_dict,inplace=True) # rename with proper suffix if possible
            
            # drop kappa columns not in full dataset
            to_drop_kappa_cols = kappa_data.columns.difference(df.columns)
            kappa_data.drop(to_drop_kappa_cols, axis=1, inplace=True)
            
            # replace with kappa data at relavent locations
            df.loc[kappa_data.index,kappa_data.columns] = kappa_data
        
        # remove 35 text columns
        txt_columns = [col_name for col_name in df.columns.astype(str) if col_name.__contains__('txt')]
        df.drop(txt_columns,axis=1,inplace=True)
        
        # remove duplicate from analysis variables
        df.drop(['clotheslining'],axis=1,inplace=True)
        
        # change some names to match dataset.py
        rename_dict = {"Race": "Race_posthoc", "PayorType": "PayorType_posthoc",\
                       "Ligamentoptions_posthoc": "LigamentInjury_posthoc"}
        df.rename(columns=rename_dict,inplace=True)
        
        # judgement call to remove any columns with date or time information
        datatime_columns = [col_name for col_name in df.columns.astype(str) if (('date' in col_name.lower()) |\
                                                                                ('time' in col_name.lower()))]
        return (df, r)

    def preprocess_data(self, cleaned_data: pd.DataFrame, datasets, **kwargs) -> pd.DataFrame:
        
        # list of categorical columns to ignore
        categorical_covariates = ['Race_posthoc','PayorType_posthoc',\
                                  'OutcomeStudySite_posthoc','OutcomeStudySiteMobility_posthoc','OutcomeStudySiteNeuro_posthoc']
        df = cleaned_data
        oss_columns = [c for c in df.columns.astype(str) if "OutcomeStudySite" in c]
        df.columns = [c +'_posthoc' if c in oss_columns else c for c in df.columns]
        
        df.rename(columns = {"AgeInYears_posthoc": "AgeInYears"}, 
          inplace = True)
        
        # add a binary outcome variable for CSI injury 
        df.loc[:,'csi_injury'] = df.index.get_level_values('control_type').map(helper.assign_binary_outcome)

        # convert numeric columns encoded as strings
        numeric_as_str_cols = ['TotalGCS', 'ModeArrival']
        for col_name in numeric_as_str_cols: # .to_numeric only works on series
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce') # coerce makes non-numeric to NA
        
        df.loc[:,'EMSArrival'] = df.loc[:,'ModeArrival'].replace([1,2],[1,0]) # did patient arrive via EMS?
        df.drop(['ModeArrival'], axis=1, inplace=True)
        
        # correct encoding for indicator of cspine immobilization
        df.loc[:,'CervicalSpineImmobilization'] = df.loc[:,'CervicalSpineImmobilization'].replace([1,2,3],[1,1,0])
        
        # change binary variable label so that 1 is abnormal condition
        df.loc[:,'NonAmbulatory'] = df.loc[:,'ambulatory'].replace([1,0],[0,1])
        df.drop(['ambulatory'], axis=1, inplace=True)
        
        # change gender in to binary indicator for male (60% majority category)
        df.loc[:,'Male'] = df.loc[:,'Gender_posthoc'].replace(['M','F','ND'],[True,False,False])
        df.drop(['Gender_posthoc'], axis=1, inplace=True)
        
        # remove ic covariates that are aggregated by other covariates
        injury_classifictation_covar = list(datasets['injuryclassification.csv'].columns.astype(str))
        # take intersection to account for txt columns already removed
        injury_classifictation_covar = list(set(injury_classifictation_covar).intersection(set(df.columns.astype(str))))
        injury_classifictation_aggregates =['CervicalSpineFractures_posthoc','LigamentInjury_posthoc',\
                                           'CervicalSpineSignalChange_posthoc']
        injury_classifictation_removed = [covar_name for covar_name in injury_classifictation_covar\
                                              if covar_name not in injury_classifictation_aggregates]
        print("IC Removed:",len(injury_classifictation_removed))
        df.drop(injury_classifictation_removed,axis=1,inplace=True)
        
        # remove radiology covariates that are aggregated by other covariates
        radiology_covar = list(datasets['radiologysite.csv'].columns.astype(str)) +\
            list(datasets['radiologyoutside.csv'].columns.astype(str))
        radiology_aggregates = ['XRays_posthoc','CTPerformed_posthoc','MRIPerformed_posthoc',\
                                'XRays_outside','CTPerformed_outside','MRIPerformed_outside']
        radiology_removed = [covar_name for covar_name in radiology_covar\
                                              if covar_name not in radiology_aggregates]
        print("Radiology Removed:",len(radiology_removed))
        df.drop(radiology_removed,axis=1,inplace=True)  
        
        # remove MOI information summarized by Leonard et al.
        moi_covar_all = list(datasets['injurymechanism.csv'].columns.astype(str))
        moi_covar_keep = ['PassRestraint','Assault','ChildAbuse','helmet','FallDownStairs']
        moi_covar_names = list(set(moi_covar_all).intersection(set(df.columns.astype(str))))
        moi_removed = [covar_name for covar_name in moi_covar_names\
                                              if covar_name not in moi_covar_keep]
        print("MOI Removed:",len(moi_removed))
        df.drop(moi_removed,axis=1,inplace=True)
        
        # drop uniformative columns which only contains a single value
        # should be 0
        no_information_columns = df.columns[df.nunique() <= 1]
        df.drop(no_information_columns, axis=1, inplace=True)
        print("# no information:", len(no_information_columns))
        
        # create one-hot encoding of AVPU data
        avpu_columns = [col for col in df.columns if 'avpu' in col.lower()]
        df[avpu_columns] = df[avpu_columns].replace('N',np.NaN).replace('Y',np.NaN)
        
        df[avpu_columns] = 'AVPU_' + df[avpu_columns].astype(str)
        avpu_one_hot = pd.get_dummies(df[avpu_columns])
        df = df.drop(avpu_columns,axis = 1)
          
        df = df.join(avpu_one_hot)
        #
        df = helper.extract_numeric_data(df,categorical_covariates=categorical_covariates)
        
        df = helper.build_binary_covariates(df)
        
        return df

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # add engineered featuures
        df = preprocessed_data     
        df = helper.rename_values(df)
        df = helper.derived_feats(df,veryyoung_age_cutoff=kwargs['veryyoung_age_cutoff'],\
                                  nonverbal_age_cutoff=kwargs['nonverbal_age_cutoff'],\
                                 young_adult_age_cutoff=kwargs['young_adult_age_cutoff'])
        
        '''
        # bin useful continuous variables age
        binning_dict = {}
        binning_dict['AgeInYears'] = (2,6,12)        
        df = helper.bin_continuous_data(df, binning_dict)
        ''' 
        return df
    
    def impute_data(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = preprocessed_data

        # impute missing binary variables with 0; this is justified because abnormal responses are encoded as 1
        # and we make a judgement call to assume that all relavent abnormal information is recorded
                      
        pd.options.mode.chained_assignment = None
        
        gcs_columns = [col for col in df.columns if 'gcs' in col.lower()]
        # GCS imputation by AlteredMentalStatus is very well justified by EDA, so we don't make it a automated JC
        # This is approved by Dr. Devlin and Dr. Kornblith
        for gcs_col in gcs_columns:
            max_gcs = df[gcs_col].max()
            df[gcs_col][(df['AlteredMentalStatus'] == 0.0) & (pd.isna(df[gcs_col]))] = max_gcs
        
        if kwargs['impute_outcomes']:
            # Judgement call to fill ~2% of units without these outcomes as normal
            df['posthoc_OutcomeStudySiteMobility'][(pd.isna(df['posthoc_OutcomeStudySiteMobility']))] = 'N'
            df['posthoc_OutcomeStudySiteNeuro'][(pd.isna(df['posthoc_OutcomeStudySiteNeuro']))] = 'NR'
        else: df = df.dropna(subset=['posthoc_OutcomeStudySiteMobility','posthoc_OutcomeStudySiteNeuro'])
                    
        # Judgement call to impute remaining ~10% of units without GCS as max e.g. 4/5/6=15
        # As with AVPU, we add an indicator of whether GCS was NA before imputation
        df['GCS_na'] = pd.isna(df['TotalGCS'].copy()).replace([True,False],[1,0])
        
        if kwargs['impute_gcs']:
            # if AMS=0, AVPU < A never occur, therefore we feeled justified imputing with max
            # AVPU A implies GCS = 15 in the complete data
            
            df[['GCSEye','MotorGCS','VerbalGCS']][(df['AlteredMentalStatus']==0)] = \
                df[['GCSEye','MotorGCS','VerbalGCS']][(df['AlteredMentalStatus']==0)]\
                .apply(lambda col: col.fillna(col.max()), axis=0)
            
            # if AMS=1, we use median imputation
            df[['GCSEye','MotorGCS','VerbalGCS']][(df['AlteredMentalStatus']==1)] = \
                df[['GCSEye','MotorGCS','VerbalGCS']][(df['AlteredMentalStatus']==1)]\
                .apply(lambda col: col.fillna(col.median()), axis=0)
            
            df['TotalGCS'] = df['GCSEye'] + df['MotorGCS'] + df['VerbalGCS'] 
            
        else: df = df.dropna(subset=['TotalGCS']) # drop any units with GCS missing, note all GCS are jointly missing
    
        df['GCSnot15'] = (df['TotalGCS'] != 15).replace([True,False],[1,0])
        df['GCSbelow9'] = (df['TotalGCS'] <= 8).replace([True,False],[1,0])
        
        pd.options.mode.chained_assignment = 'warn'

        
        '''
        # TODO
        # drop posthoc
        posthoc_columns = [col for col in df.columns if 'posthoc' in col]
        df = df.drop(posthoc_columns,axis=1).copy()
            
        # code for indicators of missing GCS
        df['GCS_NA_total'] = pd.isna(df['TotalGCS']).replace([True,False],[1,0])
        df['GCS_NA_eye'] = pd.isna(df['GCSEye']).replace([True,False],[1,0])
        df['GCS_NA_motor'] = pd.isna(df['MotorGCS']).replace([True,False],[1,0])
        df['GCS_NA_verbal'] = pd.isna(df['VerbalGCS']).replace([True,False],[1,0])
        '''
        
        for column in df.columns:
            char_column = df[column] # select column
            unique_values = pd.unique(char_column) # get unique entries
        
        # as a judgement call check, we can impute missing booleans with zero or with 
        # a bernoulli draw of their observed probability        
        
        df = helper.impute_missing_binary(df, n=kwargs['frac_missing_allowed']) 
        
        numeric_data = df.select_dtypes([np.number]) # separate data that is already numeric
        numeric_data = numeric_data.astype(float) # cast numeric data as float
        char_data = df.select_dtypes([np.object]) # gets columns encoded as strings
        
        df = pd.merge(numeric_data,char_data,how="left",left_index=True,right_index=True)
        
        return df
    
    def split_data(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Split into 3 sets: training, tuning, testing.
        Do not modify (to ensure consistent test set).
        Keep in mind any natural splits (e.g. hospitals).
        Ensure that there are positive points in all splits.

        Parameters
        ----------
        preprocessed_data
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        df_train
        df_tune
        df_test
        """
        print('split_data kwargs', kwargs)
        
        col_names = ['id','case_id','site','control_type'] + list(preprocessed_data.columns.copy())
        df_train = pd.DataFrame(columns=col_names)
        df_train = df_train.set_index(['id','case_id','site','control_type'])
        df_tune = pd.DataFrame(columns=col_names)
        df_tune = df_tune.set_index(['id','case_id','site','control_type'])
        df_test = pd.DataFrame(columns=col_names)
        df_test = df_test.set_index(['id','case_id','site','control_type'])
        
        study_site_list = [i for i in range(1,18)]
        print(kwargs['control_types'])
        selected_control_types = ['case']+kwargs['control_types']
        
        for ss in study_site_list:
            for ct in selected_control_types:
                split_subset = preprocessed_data.xs((ss, ct), level=('site','control_type'), drop_level=False) # subset to split
                
                # do the splitting below
                split_data = np.split(split_subset.sample(frac=1, random_state=42),
                                      [int(.6 * len(split_subset)), int(.8 * len(split_subset))])
                df_train = pd.concat([df_train,split_data[0]])
                df_tune = pd.concat([df_tune,split_data[1]])
                df_test = pd.concat([df_test,split_data[2]])
                
        return tuple([df_train,df_tune,df_test])

    def get_outcome_name(self) -> str:
        return 'csi_injury'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'csi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        return ['Race', 'InitHeartRate', 'InitSysBPRange']  # keys which are useful but not used for prediction

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        return {
            'clean_data': { 
                'use_kappa':[True, False, True],
            },
            'preprocess_data': {             
            },'extract_features': { 
                # some variables from `AnaylsisVariables.csv` end with a 2
                # using positive findings from field or outside hospital documentation these have 
                # the response to YES from NO or MISSING. The Leonard (2011) study considers them more robust
                # use mirror this perturbation for our own derived features
                'use_robust_av':[False, True], #TODO: refactor
                # age cutoffs choices based on rules shared by Dr. Devlin
                'veryyoung_age_cutoff':[2,1,1.5],
                'nonverbal_age_cutoff':[5,4,6],
                'young_adult_age_cutoff':[11,15],
            },
            'impute_data': { 
                # drop units with missing this percent of analysis variables or more

                'frac_missing_allowed': [0.05, 0.1],
                'impute_gcs':[True, False],
                'impute_outcomes':[True, False],
            },
            'split_data': {
                # drop cols with vals missing this percent of the time
                'control_types': [['ran','moi','ems']],
            }
        }
    
    def get_data(self, save_csvs: bool = False,
                 data_path: str = rulevetting.DATA_PATH,
                 load_csvs: bool = False,
                 run_perturbations: bool = False,
                 control_types=['ran','moi','ems'],
                 preprocess = True,
                 extract_features = True,
                 impute = True) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """Runs all the processing and returns the data.
        This method does not need to be overriden.

        Params
        ------
        save_csvs: bool, optional
            Whether to save csv files of the processed data
        data_path: str, optional
            Path to all data
        load_csvs: bool, optional
            Whether to skip all processing and load data directly from csvs
        run_perturbations: bool, optional
            Whether to run / save data pipeline for all combinations of judgement calls
        control_types: list of str, optional
            Which control types (Random, Mechanism of Injury, EMS) to include
        Returns
        -------
        df_train
        df_tune
        df_test
        """
        PROCESSED_PATH = oj(data_path, self.get_dataset_id(), 'processed')

        if load_csvs:
            return tuple([pd.read_csv(oj(PROCESSED_PATH, s), index_col=0)
                          for s in ['train.csv', 'tune.csv', 'test.csv']])
        np.random.seed(0)
        random.seed(0)
        CACHE_PATH = oj(data_path, 'joblib_cache')
        cache = Memory(CACHE_PATH, verbose=0).cache
        kwargs = self.get_judgement_calls_dictionary()
        default_kwargs = {}
        for key in kwargs.keys():
            func_kwargs = kwargs[key]
            default_kwargs[key] = {k: func_kwargs[k][0]  # first arg in each list is default
                                   for k in func_kwargs.keys()}
        
        data_tuple = cache(self.clean_data)(data_path=data_path, **default_kwargs['clean_data'])
        if preprocess:
            data = cache(self.preprocess_data)(data_tuple[0], data_tuple[1], **default_kwargs['preprocess_data'])
        if preprocess and extract_features:
            data = cache(self.extract_features)(data, **default_kwargs['extract_features'])
        if preprocess and extract_features and impute:
            data = cache(self.impute_data)(data, **default_kwargs['impute_data'])
        
        df_train, df_tune, df_test = cache(self.split_data)(data, **{'control_types': control_types})

        return df_train, df_tune, df_test


if __name__ == '__main__':
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=True)
    print('successfuly processed data\nshapes:',
          df_train.shape, df_tune.shape, df_test.shape,
          '\nfeatures:', list(df_train.columns))