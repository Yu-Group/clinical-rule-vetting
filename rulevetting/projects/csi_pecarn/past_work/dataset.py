from os.path import join as oj

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict

import rulevetting
import rulevetting.api.util
from rulevetting.projects.iai_pecarn import helper
from rulevetting.templates.dataset import DatasetTemplate


class Dataset(DatasetTemplate):
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        # all the fnames to be loaded and searched over
        
        fnames = sorted([
            fname for fname in os.listdir(raw_data_path)
            if 'csv' in fname
            #   and not 'radiology' in fname
            #   and not 'kappa' in fname
            ])

        # read through each fname and save into the r dictionary
        r = {}
        print('read all the csvs...', fnames)
        if len(fnames) == 0:
            print('no csvs found in path', raw_data_path)
        for fname in tqdm(fnames):
            df = pd.read_csv(oj(raw_data_path, fname), encoding="ISO-8859-1")
            df.rename(columns={'site': 'SiteID'}, inplace=True)
            df.rename(columns={'SITE': 'SiteID'}, inplace=True)
            df.rename(columns={'caseid': 'CaseID'}, inplace=True)
            df.rename(columns={'controltype': 'ControlType'}, inplace=True)
            df.rename(columns={'studysubjectid': 'SubjectID'}, inplace=True)
            df.rename(columns={'StudySubjectID': 'SubjectID'}, inplace=True)
            assert ('SiteID' in df.keys())
            assert ('CaseID' in df.keys())
            assert ('ControlType' in df.keys())
            assert ('SubjectID' in df.keys())
            r[fname] = df

        # loop over the relevant forms and merge into one big df
        fnames_small = [fname for fname in fnames
                        if not 'radiology' in fname
                            and not 'kappa' in fname
                            and not 'injuryclassification' in fname
                            and not 'outside' in fname
                            and not 'onfield' in fname]

        df_features = r[fnames_small[0]]
        print('merge all the dfs...')
        for i, fname in tqdm(enumerate(fnames_small)):
            df2 = r[fname].copy()
            
            # if subj has multiple entries, only keep first
            df2 = df2.drop_duplicates(subset=['SubjectID'], keep='last')
            
            # don't save duplicate columns
            df_features = df_features.set_index('SubjectID').combine_first(df2.set_index('SubjectID')).reset_index()

        df_outcomes = helper.get_outcomes(raw_data_path)

        df = pd.merge(df_features, df_outcomes, on='id', how='left')
        df = helper.rename_values(df)  # rename the features by their meaning

        return df

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # drop cols with vals missing this percent of the time
        df = cleaned_data.dropna(axis=1, thresh=(1 - kwargs['frac_missing_allowed']) * cleaned_data.shape[0])

        # impute missing values
        # fill in values for some vars from unknown -> None
        df.loc[df['AbdomenTender'].isin(['no', 'unknown']), 'AbdTenderDegree'] = 'None'

        # pandas impute missing values with median
        df = df.fillna(df.median())
        df.GCSScore = df.GCSScore.fillna(df.GCSScore.median())

        df['outcome'] = df[self.get_outcome_name()]

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

    def get_outcome_name(self) -> str:
        return 'iai_intervention'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'iai_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        return ['Race', 'InitHeartRate', 'InitSysBPRange']  # keys which are useful but not used for prediction

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        return {
            'clean_data': {},
            'preprocess_data': {
                # drop cols with vals missing this percent of the time
                'frac_missing_allowed': [0.05, 0.10],
            },
            'extract_features': {
                # whether to drop columns with suffix _no
                'drop_negative_columns': [False],  # default value comes first
            },
        }


if __name__ == '__main__':
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=True)
    print('successfuly processed data\nshapes:',
          df_train.shape, df_tune.shape, df_test.shape,
          '\nfeatures:', list(df_train.columns))
