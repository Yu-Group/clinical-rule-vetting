import os
import random
from abc import abstractmethod
from os.path import join as oj
from typing import Dict
from tqdm import tqdm

import numpy as np
import pandas as pd
from joblib import Memory

import rulevetting
from rulevetting.projects import one_hot_encode_df
from rulevetting.projects.tbi_pecarn import helper
from vflow import init_args, vset, build_vset


class Dataset:
    """All functions take **kwargs, so you can specify any judgement calls you aren't sure about with a kwarg flag. Please refrain from shuffling / reordering the data in any of these functions, to ensure a consistent test set.
    """

    @abstractmethod
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        """
        Convert the raw data files into a pandas dataframe.
        Dataframe keys should be reasonable (lowercase, underscore-separated).
        Data types should be reasonable.

        Params
        ------
        data_path: str, optional
            Path to all data files
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        cleaned_data: pd.DataFrame
        """
        # get the file path to the raw data frame
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        # all the file names are loaded and searched over
        fnames = sorted([
            fname for fname in os.listdir(raw_data_path)
            if 'csv' in fname]) 

        # take each csv path, read into dataframe, and add to dictionary r
        r = {}
        print('Reading the raw CSVs...', fnames)
        if len(fnames) == 0:
            print('No CSVs in path', raw_data_path)
        for fname in tqdm(fnames):
            df = pd.read_csv(oj(raw_data_path, fname), encoding="ISO-8859-1")
            df.rename(columns={'PatNum': 'id'}, inplace=True)
            df.rename(columns={'patnum': 'id'}, inplace=True)
            assert ('id' in df.keys())

            # if subj has multiple entries, only keep first
            df = df.drop_duplicates(subset=['id'], keep='last')

            # apply helper functions to make categorical vars and rename values
            if fname == 'TBI PUD 10-08-2013.csv':
                df = helper.rename_tbi_pud(df)
            if fname == 'TBI PUD Neuro.csv':
                df = helper.rename_tbi_neuro(df)
            r[fname] = df

        # just looking at the first dataframe with pre-ct variables
        df = r['TBI PUD 10-08-2013.csv']
        cleaned_data = df.replace('nan', np.nan)
        cleaned_data['AgeTwoPlus'] -= 1

        return cleaned_data

    @abstractmethod
    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Preprocess the data.
        Impute missing values.
        Scale/transform values.
        Should put the prediction target in a column named "outcome"

        Parameters
        ----------
        cleaned_data: pd.DataFrame
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        preprocessed_data: pd.DataFrame
        """
        
        # # infer missing PosIntFinal from the other outcome columns
        # def infer_missing_outcome(row):
        #     outcome = 'Unknown'
        #     # look at known outcome columns to infer outcome
        #     not_missing = [data for data in row if data != 'Unknown']

        #     # if all values that are known give the same answer, use that as the outcome
        #     # If all values are present (no missings)
        #     if len(not_missing) == len(row) and not_missing.count(not_missing[0]) == len(not_missing):
        #         outcome = not_missing[0]
        #     return outcome
        
        # judgement call - we infer missing outcomes based on other outcome variables - hosphead, intub, ...
        if kwargs['infer_outcome']:
            cleaned_data = cleaned_data.loc[
                ~cleaned_data['PosIntFinal'].isna() | ~cleaned_data[['HospHeadPosCT', 'Intub24Head', 'Neurosurgery', 'DeathTBI']].isna().any(axis=1)]
            
            cleaned_data.loc[cleaned_data['PosIntFinal'].isna(), 'PosIntFinal'] = (cleaned_data.loc[
                cleaned_data['PosIntFinal'].isna(), ['HospHeadPosCT', 'Intub24Head', 'Neurosurgery', 'DeathTBI']].sum(axis=1) >= 1).astype(int)
        
        # judgement call - we drop patients with gcs <14 and thus gcs scores
        if kwargs['drop_low_gcs']:
            cleaned_data = cleaned_data.loc[cleaned_data['GCSTotal'] >= 14, :]
            cleaned_data = cleaned_data.drop(columns = ['GCSTotal', 'GCSGroup'])
            gcs_vars = ['GCSEye', 'GCSMotor', 'GCSVerbal']
            cleaned_data = cleaned_data.drop(columns=gcs_vars)

        if not kwargs['propensity']:

            # judgement call - impute unknowns with mode/mean/or drop patient
            if 'impute_unknowns' in kwargs:
                for col in cleaned_data.columns:
                    if kwargs['impute_unknowns'] == 'mode':
                        cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])

                    if kwargs['impute_unknowns'] == 'drop':
                        cleaned_data = cleaned_data[cleaned_data[col].isna()]
                
            # judgement call - impute features that have descriptions about dealing with not applicables
            not_applicable_doc_feats = ['SeizOccur', 'VomitNbr', 'VomitStart', 'VomitLast', 'AMSAgitated', 'AMSSleep',
                                        'AMSSlow', 'AMSRepeat', 'AMSOth', 'SFxBasHem', 'SFxBasOto', 'SFxBasPer', 
                                        'SFxBasRet', 'SFxBasRhi', 'ClavFace', 'ClavNeck', 'ClavFro', 'ClavOcc', 
                                        'ClavPar', 'ClavTem', 'NeuroD', 'NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial',
                                        'NeuroDReflex', 'NeuroDOth', 'OSIExtremity', 'OSICut', 'OSICspine', 'OSIFlank',
                                        'OSIAbdomen', 'OSIPelvis', 'OSIOth', 'High_impact_InjSev']
            if kwargs['impute_not_applicables']:
                cleaned_data.loc[:, not_applicable_doc_feats] = cleaned_data.loc[:, not_applicable_doc_feats].replace({92: 0})

        # dropping variables that do not influence the doctors decision
        other_vars = ['EmplType', 'Certification', 'Race']
        cleaned_data = cleaned_data.drop(columns=other_vars)
            
        # renaming our target variable
        cleaned_data.rename(columns = {'PosIntFinal': 'outcome'}, inplace=True)
        
        # removing post-ct variables that aren't the outcome
        cleaned_data = cleaned_data.drop(columns=self.get_post_ct_names())
        
        # dropping variables that do have high fraction of missing values
        hi_missing_vars = ['Dizzy', 'Ethnicity']
        cleaned_data = cleaned_data.drop(columns=hi_missing_vars)
     
        # IF keep years, don't drop it
        if not kwargs["keep_years"]:
            cleaned_data = cleaned_data.drop(columns=['AgeInMonth', 'AgeInYears'])
        
        # remapping binary variables
        # bool_cols = [col for col in cleaned_data if np.isin(cleaned_data[col].unique(), ['No', 'Yes', ]).all()]
        # cleaned_data.loc[:, bool_cols] = cleaned_data.loc[:, bool_cols].replace({'No': 0, 'Yes': 1})
    
        # gender has to be remapped - if we actually use it
        # cleaned_data['Gender'] = cleaned_data['Gender'].map({'Male': 0, 'Female': 1})
    
        # id isn't necessary post eda
        preprocessed_data = cleaned_data.drop(columns=['id'])

        # one-hot encode categorical vars w/ >2 unique values
        numeric_cols = ['AgeInMonth', 'AgeinYears']
        preprocessed_data = one_hot_encode_df(preprocessed_data, numeric_cols)

        preprocessed_data.insert(
            len(preprocessed_data.columns) - 1, 'outcome', preprocessed_data.pop('outcome'))
        
        return preprocessed_data.astype(np.float32)
    
    @abstractmethod
    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extract features from preprocessed data
        All features should be binary


        Parameters
        ----------
        preprocessed_data: pd.DataFrame
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls,
            extract_features key should be a list of features/strings

        Returns
        -------
        extracted_features: pd.DataFrame
        """
        extracted_features = preprocessed_data
        if 'extract_features' in kwargs and len(kwargs['extract_features']) > 0:
            extracted_features = preprocessed_data[kwargs['extract_features']]
            
        return extracted_features

    def split_data(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
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
        return tuple(np.split(
            preprocessed_data.sample(frac=1, random_state=42),
            [int(.6 * len(preprocessed_data)),  # 60% train
             int(.8 * len(preprocessed_data))]  # 20% tune, 20% test
        ))
    
    @abstractmethod
    def get_data_split(self, pre_data: pd.DataFrame, simple = False, young = True, old = True, verbal_split = False, nonverbal = True, verbal = True) :
        """Retrieve preprocessed data and returns the requested dataset (train, tune, test)
        
        Parameters
        ----------
        pre_data : pd.DataFrame  - Preprocessed data
        simple : True or False   - Only contains simple variables if True, contains all variables if False
        young  : True or False   - Data contains age < 2
        old    : True or False   - Data contains age > 2

        Returns
        -------
        X_train : pd.DataFrame     y_train : np.array
        X_tune : pd.DataFrame      y_tune : np.array
        X_test : pd.DataFrame      y_test : np.array
        """
    
        outcome_def = self.get_outcome_name()
        simple_var_list = self.get_simple_var_list()
        
        if simple:
            pre_data = pre_data[simple_var_list]
        if not verbal_split:
            if young and not old: 
                pre_data = pre_data.loc[pre_data['AgeTwoPlus'] == 1.0, :]
                pre_data = pre_data.drop(columns = ['AgeTwoPlus'])
            if old and not young:
                pre_data = pre_data.loc[pre_data['AgeTwoPlus'] == 2.0, :]
                pre_data = pre_data.drop(columns = ['AgeTwoPlus'])
        elif verbal_split:
            if verbal and not nonverbal:
                pre_data = pre_data.loc[pre_data['HA_verb'] != 91, :]
                pre_data = pre_data.drop(columns = ['HA_verb'])
            if nonverbal and not verbal:
                pre_data = pre_data.loc[pre_data['HA_verb'] == 91, :]
                pre_data = pre_data.drop(columns = ['HA_verb'])   
        
        df_train, df_tune, df_test = self.split_data(pre_data)
        X_train = df_train.drop(columns=outcome_def)
        y_train = df_train[outcome_def].values
        X_tune = df_tune.drop(columns=outcome_def)
        y_tune = df_tune[outcome_def].values
        X_test = df_test.drop(columns=outcome_def)
        y_test = df_test[outcome_def].values

        return X_train, y_train, X_tune, y_tune, X_test, y_test

    @abstractmethod
    def get_outcome_name(self) -> str:
        return 'outcome'  # return the name of the outcome we are predicting

    @abstractmethod
    def get_post_ct_names(self) -> list:
        tbi_on_ct = [f'Finding{i}' for i in range(1, 15)] + [f'Finding{i}' for i in range(20, 24)] + ['PosCT']
        ctform_vars = ['CTForm1', 'IndAge', 'IndAmnesia', 'IndAMS', 'IndClinSFx',
                       'IndHA', 'IndHema', 'IndLOC', 'IndMech', 'IndNeuroD',
                       'IndRqstMD', 'IndRqstParent', 'IndRqstTrauma', 'IndSeiz', 'IndVomit',
                       'IndXraySFx', 'IndOth', 'CTSed', 'CTSedAgitate', 'CTSedAge', 
                       'CTSedRqst', 'CTSedOth']
        outcome_vars = ['HospHeadPosCT', 'DeathTBI', 'HospHead', 'Intub24Head', 'Neurosurgery']
        other_vars = ['CTDone', 'EDCT', 'EDDisposition', 'Observed']
        
        return  tbi_on_ct + ctform_vars + outcome_vars + other_vars # return name of post ct vars that aren't the outcome

    @abstractmethod
    def get_simple_var_list(self) -> list:
        return ['InjuryMech_Assault', 'InjuryMech_Bicyclist struck by automobile',
           'InjuryMech_Bike collision/fall', 'InjuryMech_Fall down stairs',
           'InjuryMech_Fall from an elevation',
           'InjuryMech_Fall to ground standing/walking/running',
           'InjuryMech_Motor vehicle collision',
           'InjuryMech_Object struck head - accidental',
           'InjuryMech_Other mechanism', 'InjuryMech_Other wheeled crash',
           'InjuryMech_Pedestrian struck by moving vehicle', 'InjuryMech_Sports',
           'InjuryMech_Walked/ran into stationary object',
           'High_impact_InjSev_High', 'High_impact_InjSev_Low',
           'High_impact_InjSev_Moderate', 'Amnesia_verb_No', 
           'Amnesia_verb_Pre/Non-verbal', 'Amnesia_verb_Yes',
           'LOCSeparate_No', 'LOCSeparate_Suspected', 'LOCSeparate_Yes', 
           'Seiz', 'ActNorm', 'HA_verb_No', 'HA_verb_Pre/Non-verbal', 'HA_verb_Yes',
            'Vomit', 'Intubated', 'Paralyzed', 'Sedated',
            'AMS', 'SFxPalp_No', 'SFxPalp_Unclear', 'SFxPalp_Yes',
           'FontBulg', 'Hema', 'Clav', 'NeuroD', 'OSI', 'Drugs', 'AgeTwoPlus', 'outcome']
    
    @abstractmethod
    def get_dataset_id(self) -> str:
        return 'tbi_pecarn'  # return the name of the dataset id

    @abstractmethod
    def get_meta_keys(self) -> list:
        """Return list of keys which should not be used in fitting but are still useful for analysis
        """
        return ['Ethnicity', 'Gender', 'Race']

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        """Return dictionary of keyword arguments for each function in the dataset class.
        Each key should be a string with the name of the arg.
        Each value should be a list of values, with the default value coming first.
        """
        return {
            'clean_data': {
                'propensity': [True, False]
            },
            'preprocess_data': {
                'infer_outcome': [True, False],
                'keep_years': [True, False],
                'drop_low_gcs': [True, False],
                'impute_unknowns': ['mode', 'drop'],
                'impute_not_applicables': [True, False],
                'propensity': [True, False]
            },
            'extract_features': {}
        }

    def get_data(self, save_csvs: bool = False,
                 data_path: str = rulevetting.DATA_PATH,
                 load_csvs: bool = False,
                 simple: bool = True,
                 young: bool = True,
                 old: bool = True,
                 verbal_split: bool = False,
                 nonverbal: bool = True,
                 verbal: bool = True,
                 run_perturbations: bool = False, **kwargs) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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
        simple : bool, optional
            contains simple variables if True, contains all variables if False
        young : bool, optional
            use data for patients with age < 2 or not include them
        old : bool, optional
            use data for patients with age >= 2 or not include them
        verbal_split: bool, optional
            Whether to instead split the data by verbal/not
        run_perturbations: bool, optional
            Whether to run / save data pipeline for all combinations of judgement calls

        Returns
        -------
        df_train
        df_tune
        df_test
        """
        outcome_def = self.get_outcome_name()
        simple_var_list = self.get_simple_var_list()
        
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

        print('kwargs', default_kwargs)
        if not run_perturbations:
            cleaned_data = self.clean_data(data_path=data_path, **default_kwargs['clean_data'], **kwargs)
            preprocessed_data = self.preprocess_data(cleaned_data, **default_kwargs['preprocess_data'], **kwargs)
            extracted_features = self.extract_features(preprocessed_data, **default_kwargs['extract_features'], **kwargs)
            pre_data = extracted_features
            
            if simple:
                pre_data = pre_data[simple_var_list]
            if not verbal_split:
                if young and not old:
                    pre_data = pre_data.loc[pre_data['AgeTwoPlus'] == 0, :]
                    pre_data = pre_data.drop(columns = ['AgeTwoPlus'])
                if old and not young:
                    pre_data = pre_data.loc[pre_data['AgeTwoPlus'] == 1, :]
                    pre_data = pre_data.drop(columns = ['AgeTwoPlus'])
            elif verbal_split:
                if verbal and not nonverbal:
                    pre_data = pre_data.loc[pre_data['HA_verb'] != 91, :]
                    pre_data = pre_data.drop(columns = ['HA_verb'])
                if nonverbal and not verbal:
                    pre_data = pre_data.loc[pre_data['HA_verb'] == 91, :]
                    pre_data = pre_data.drop(columns = ['HA_verb'])             

            df_train, df_tune, df_test = self.split_data(pre_data)
            
        elif run_perturbations:
            data_path_arg = init_args([data_path], names=['data_path'])[0]
            clean_set = build_vset('clean_data', self.clean_data, param_dict=kwargs['clean_data'], cache_dir=CACHE_PATH)
            cleaned_data = clean_set(data_path_arg)
            preprocess_set = build_vset('preprocess_data', self.preprocess_data, param_dict=kwargs['preprocess_data'],
                                        cache_dir=CACHE_PATH)
            preprocessed_data = preprocess_set(cleaned_data)
            extract_set = build_vset('extract_features', self.extract_features, param_dict=kwargs['extract_features'],
                                     cache_dir=CACHE_PATH)
            extracted_features = extract_set(preprocessed_data)
            split_data = vset('split_data', modules=[self.split_data])
            dfs = split_data(extracted_features)
        if save_csvs:
            os.makedirs(PROCESSED_PATH, exist_ok=True)

            if not run_perturbations:
                for df, fname in zip([df_train, df_tune, df_test],
                                     ['train.csv', 'tune.csv', 'test.csv']):
                    meta_keys = rulevetting.api.util.get_feat_names_from_base_feats(df.keys(), self.get_meta_keys())
                    df.loc[:, meta_keys].to_csv(oj(PROCESSED_PATH, f'meta_{fname}'))
                    df.drop(columns=meta_keys).to_csv(oj(PROCESSED_PATH, fname))
            if run_perturbations:
                for k in dfs.keys():
                    if isinstance(k, tuple):
                        os.makedirs(oj(PROCESSED_PATH, 'perturbed_data'), exist_ok=True)
                        perturbation_name = str(k).replace(', ', '_').replace('(', '').replace(')', '')
                        perturbed_path = oj(PROCESSED_PATH, 'perturbed_data', perturbation_name)
                        os.makedirs(perturbed_path, exist_ok=True)
                        for i, fname in enumerate(['train.csv', 'tune.csv', 'test.csv']):
                            df = dfs[k][i]
                            meta_keys = rulevetting.api.util.get_feat_names_from_base_feats(df.keys(),
                                                                                            self.get_meta_keys())
                            df.loc[:, meta_keys].to_csv(oj(perturbed_path, f'meta_{fname}'))
                            df.drop(columns=meta_keys).to_csv(oj(perturbed_path, fname))
                return dfs[list(dfs.keys())[0]]

        return df_train, df_tune, df_test
