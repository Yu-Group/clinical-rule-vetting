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
        
        # read through each fname and save into the r dictionary
        r = {}
        print('read all the csvs...\n', fnames)
        if len(fnames) == 0:
            print('no csvs found in path', raw_data_path)
        
        # replace studysubjectid cases with id
        for fname in tqdm(fnames):
            df = pd.read_csv(oj(raw_data_path, fname), encoding="ISO-8859-1")
            df.rename(columns={'StudySubjectID': 'id'}, inplace=True)
            df.rename(columns={'studysubjectid': 'id'}, inplace=True)
            assert ('id' in df.keys())
            r[fname] = df
            
            df.columns = [re.sub('SITE','site',x) for x in df.columns]
            df.columns = [re.sub('CaseID','case_id',x) for x in df.columns]
            df.columns = [re.sub('CSpine','CervicalSpine',x) for x in df.columns]
            df.columns = [re.sub('ControlType','control_type',x,flags=re.IGNORECASE) for x in df.columns]

        # Get filenames we consider in our covariate analysis
        # We do not consider radiology data or injury classification because this data is not
        # available at decision time in the ED.
        fnames_small = [fname for fname in fnames
                        if not 'radiology' in fname
                        and not 'injuryclassification' in fname
                        and not 'kappa' in fname]
        
        df_features = r[fnames[0]] # keep `site`, `case id`, and `control type` covar from first df
        
        print('merging all of the dfs...')
        for i, fname in tqdm(enumerate(fnames_small)):
            df2 = r[fname].copy()

            # if subj has multiple entries, only keep first
            # there are no duplicates in this dataset but we keep this code for robustness to future data
            df2 = df2.drop_duplicates(subset=['id'], keep='last')
            df2_features = df2.iloc[:,3:] # ignore `site`, `case id` and `contol type`
            # don't save duplicate columns; the above line catches errors from different cased var names
            df_features = df_features.set_index('id').combine_first(df2_features.set_index('id')).reset_index()
        
        # set_index commands merge but do not properly set index
        # use a multi-indexing for easily work with binary features
        df_features = df_features.set_index(['id','case_id','site','control_type']) # use a multiIndex
        
        # remove variables collected after the intervention
        posthoc_columns = df_features.columns[df_features.columns.str.startswith('IntervFor')].union( \
            df_features.columns[df_features.columns.str.contains('LongTerm')]).union( \
            df_features.columns[df_features.columns.str.contains('Outcome')])
        df_features.drop(posthoc_columns, axis=1, inplace=True)
        
        # change binary variable label so that 1 is negative result
        df_features.loc[:,'NonAmbulatory'] = df_features.loc[:,'ambulatory'].replace([1,0],[0,1])
        df_features.drop(['ambulatory'], axis=1, inplace=True)
        
        # judgement call to get analysis variable columns that end with 2 (more robust)
        # first standardize column names
        df_features.columns = [re.sub('subinj_','SubInj_',x) for x in df_features.columns]
        # then get relavent columns
        robust_av_columns = df_features.columns[df_features.columns.str.endswith('2')]\
            .drop(['PtTenderNeckLevelC2','ICD9Location2','ICD9MechInjury2']) # endswith 2 regex is too strong
        nonrobust_av_columns = [col_name[:-1] for col_name in robust_av_columns] 
        # drop columns for jdugement call
        if kwargs['use_robust_av']: df_features.drop(nonrobust_av_columns, axis=1, inplace=True)
        else: df_features.drop(robust_av_columns, axis=1, inplace=True)
            
        # judgement call to use kappa variables where appropriate
        if kwargs['use_kappa']:
            kappa_data = r['kappa.csv']
            kappa_data = kappa_data.set_index('id')
            # drop kappa columns not in full dataset
            to_drop_kappa_cols = kappa_data.columns.difference(df_features.columns)
            kappa_data.drop(to_drop_kappa_cols, axis=1, inplace=True)
            # replace with kappa data at relavent locations
            df_features.loc[kappa_data.index,kappa_data.columns] = kappa_data

        return df_features

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        # create one-hot encoding of AVPU data
        cleaned_data['AVPUDetails'] = cleaned_data['AVPUDetails'].replace('N',np.NaN)
        cleaned_data['AVPUDetails'] = 'AVPU_' + cleaned_data['AVPUDetails'].astype(str)
        avpu_one_hot = pd.get_dummies(cleaned_data['AVPUDetails'])
        cleaned_data = cleaned_data.drop(['AVPU','AVPUDetails'],axis = 1)
        cleaned_data = cleaned_data.join(avpu_one_hot)
        
        df = helper.extract_numeric_data(cleaned_data)
                
        # impute missing values
        df = helper.impute_missing(df, n=kwargs['frac_missing_allowed']) # drop some observations and impute other missing values 
        # drop cols with vals missing this percent of the time
        #df = df.dropna(axis=1, thresh=(1 - kwargs['frac_missing_allowed']) * cleaned_data.shape[0])

        # add a binary outcome variable for CSI injury 
        df.loc[:,'csi_injury'] = df.index.get_level_values('control_type').map(helper.assign_binary_outcome)
        
        # drop uniformative columns which only contains a single value
        no_information_columns = df.columns[df.nunique() <= 1]
        df.drop(no_information_columns, axis=1, inplace=True)
        
        # bin useful continuous variables Age and (perhaps) EMSArrivalTime
        # TODO: make cutoffs a judgement call
        binning_dict = {}
        binning_dict['AgeInYears'] = (2,5,13)        
        df = helper.bin_continuous_data(df, binning_dict)
        
        return df

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # add engineered featuures
        df = helper.derived_feats(preprocessed_data)

        # convert feats to dummy
        df = pd.get_dummies(df, dummy_na=True)  # treat na as a separate category

        # remove any col that is all 0s
        df = df.loc[:, (df != 0).any(axis=0)]

        # remove the _no columns
        if kwargs['drop_negative_columns']:
            df.drop([k for k in df.keys() if k.endswith('_no')], inplace=True)

        # narrow to good keys
        feat_names = [k for k in df.keys()  # features to use
                      if not 'iai' in k.lower()]
        base_feat_names = []
        base_feat_names += ['AbdDistention', 'AbdTenderDegree', 'AbdTrauma', 'AbdTrauma_or_SeatBeltSign',
                            'AbdomenPain', 'Costal', 'DecrBreathSound', 'DistractingPain',
                            'FemurFracture', 'GCSScore', 'Hypotension', 'LtCostalTender',
                            'MOI', 'RtCostalTender', 'SeatBeltSign', 'ThoracicTender',
                            'ThoracicTrauma', 'VomitWretch', 'Age', 'Sex']
        base_feat_names += self.get_meta_keys()
        feats = rulevetting.api.util.get_feat_names_from_base_feats(feat_names,
                                                                    base_feat_names=base_feat_names) + ['outcome']
        return df[feats]
    
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

    def get_outcome_name(self) -> str:
        return 'csi_injury'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'csi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        return ['Race', 'InitHeartRate', 'InitSysBPRange']  # keys which are useful but not used for prediction

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        return {
            'clean_data': {
                # Kappa data is re-abstracted by an indepedent doctor on a subset of `ClinicalPresentationSite.csv` data 
                # If true, use this data for relavent units
                'use_kappa':[False, True],
                
                # some variables from `AnaylsisVariables.csv` end with a 2
                # using positive findings from field or outside hospital documentation these have 
                # the response to YES from NO or MISSING. The Leonard (2011) study considers them more robust
                'use_robust_av':[True, False],
            },
            'preprocess_data': {
                # drop cols with vals missing this percent of the time
                'frac_missing_allowed': [0.05, 0.10],
            },
            'extract_features': {
                # whether to drop columns with suffix _no
                'drop_negative_columns': [False],  # default value comes first
            }
        }
    
    def get_data(self, save_csvs: bool = False,
                 data_path: str = rulevetting.DATA_PATH,
                 load_csvs: bool = False,
                 run_perturbations: bool = False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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

        print('kwargs', default_kwargs)
        if not run_perturbations:
            cleaned_data = cache(self.clean_data)(data_path=data_path, **default_kwargs['clean_data'])
            preprocessed_data = cache(self.preprocess_data)(cleaned_data, **default_kwargs['preprocess_data'])
            #extracted_features = cache(self.extract_features)(preprocessed_data, **default_kwargs['extract_features'])
            #df_train, df_tune, df_test = cache(self.split_data)(extracted_features)
        elif run_perturbations:
            data_path_arg = init_args([data_path], names=['data_path'])[0]
            clean_set = build_Vset('clean_data', self.clean_data, param_dict=kwargs['clean_data'], cache_dir=CACHE_PATH)
            cleaned_data = clean_set(data_path_arg)
            preprocess_set = build_Vset('preprocess_data', self.preprocess_data, param_dict=kwargs['preprocess_data'],
                                        cache_dir=CACHE_PATH)
            preprocessed_data = preprocess_set(cleaned_data)
            extract_set = build_Vset('extract_features', self.extract_features, param_dict=kwargs['extract_features'],
                                     cache_dir=CACHE_PATH)
            extracted_features = extract_set(preprocessed_data)
            split_data = Vset('split_data', modules=[self.split_data])
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

        return preprocessed_data #df_train, df_tune, df_test


if __name__ == '__main__':
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=True)
    print('successfuly processed data\nshapes:',
          df_train.shape, df_tune.shape, df_test.shape,
          '\nfeatures:', list(df_train.columns))
