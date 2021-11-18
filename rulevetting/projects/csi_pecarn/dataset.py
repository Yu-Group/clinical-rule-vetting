import glob
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
        fnames = sorted(glob.glob(f'{raw_data_path}/*'))
        dfs = [pd.read_csv(fname) for fname in fnames]

        return dfs[0]

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # drop cols with vals missing this percent of the time
        # df = cleaned_data.dropna(axis=1, thresh=(1 - kwargs['frac_missing_allowed']) * cleaned_data.shape[0])

        # impute missing values
        # fill in values for some vars from unknown -> None
        df = cleaned_data.dropna(axis=0)

        # pandas impute missing values with median
        # df = df.fillna(df.median())
        # df.GCSScore = df.GCSScore.fillna(df.GCSScore.median())

        df.loc[:, 'outcome'] = (df['ControlType'] == 'case').astype(int)

        return df

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # add engineered featuures
        # df = helper.derived_feats(preprocessed_data)

        # convert feats to dummy
        # df = pd.get_dummies(df, dummy_na=True)  # treat na as a separate category

        # remove any col that is all 0s
        df = preprocessed_data.loc[:, (preprocessed_data != 0).any(axis=0)]

        # remove the _no columns
        # if kwargs['drop_negative_columns']:
         #    df.drop([k for k in df.keys() if k.endswith('_no')], inplace=True)

        # remove site, case ID, subject ID, control type
        feats = df.keys()[4:]

        return df[feats]

    def get_outcome_name(self) -> str:
        return 'csi'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'csi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        return ['SITE', 'CaseID']  # keys which are useful but not used for prediction

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
