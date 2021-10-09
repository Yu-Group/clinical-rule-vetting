from os.path import join as oj

import numpy as np
import os
import pandas as pd
from tqdm import tqdm

import mrules
from mrules.projects.iai_pecarn import helper
from mrules.util.dataset import MDataset


class Dataset(MDataset):

    def clean_data(self) -> pd.DataFrame:
        """
        Convert the raw data files into a pandas dataframe.
        Dataframe keys should be reasonable (lowercase, underscore-separated)

        Returns
        -------
        cleaned_data: pd.DataFrame
        """

        RAW_DATA_PATH = oj(mrules.DATA_PATH, 'iai_pecarn', 'raw')
        os.makedirs(RAW_DATA_PATH, exist_ok=True)

        # all the fnames to be loaded and searched over
        fnames = sorted([
            fname for fname in os.listdir(RAW_DATA_PATH)
            if 'csv' in fname
               and not 'formats' in fname
               and not 'form6' in fname])  # remove outcome

        # read through each fname and save into the r dictionary
        r = {}
        print('read all the csvs...', fnames)
        for fname in tqdm(fnames):
            df = pd.read_csv(oj(RAW_DATA_PATH, fname), encoding="ISO-8859-1")
            df.rename(columns={'SubjectID': 'id'}, inplace=True)
            df.rename(columns={'subjectid': 'id'}, inplace=True)
            assert ('id' in df.keys())
            r[fname] = df

        # loop over the relevant forms and merge into one big df
        fnames_small = [fname for fname in fnames
                        if 'form1' in fname
                        or 'form2' in fname
                        or 'form4' in fname
                        or 'form5' in fname
                        or 'form7' in fname
                        ]
        df_features = r[fnames[0]]
        print('merge all the dfs...')
        for i, fname in tqdm(enumerate(fnames_small)):
            df2 = r[fname].copy()

            # if subj has multiple entries, only keep first
            df2 = df2.drop_duplicates(subset=['id'], keep='last')

            # don't save duplicate columns
            df_features = df_features.set_index('id').combine_first(df2.set_index('id')).reset_index()

        df_outcomes = helper.get_outcomes(RAW_DATA_PATH)

        df = pd.merge(df_features, df_outcomes, on='id', how='left')
        df = helper.rename_values(df)  # rename the features by their meaning

        return df

    def preprocess_data(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data.
        Impute missing values.
        Should put the prediction target in a column named "outcome"

        Parameters
        ----------
        cleaned_data: pd.DataFrame

        Returns
        -------
        preprocessed_data: pd.DataFrame
        """

        # drop cols with vals missing this percent of the time
        FRAC_MISSING_ALLOWED = 0.05
        NUM_PATIENTS = 12044
        df = cleaned_data.dropna(axis=1, thresh=(1 - FRAC_MISSING_ALLOWED) * NUM_PATIENTS)

        # impute missing values
        # fill in values for some vars from unknown -> None
        df.loc[df['AbdomenTender'].isin(['no', 'unknown']), 'AbdTenderDegree'] = 'None'

        # pandas impute missing values with median
        df = df.fillna(df.median())
        df.GCSScore = df.GCSScore.fillna(df.GCSScore.median())

        df['outcome'] = df[self.get_outcome_name()]

        return df

    def extract_features(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from preprocessed data
        All features should be binary


        Parameters
        ----------
        preprocessed_data

        Returns
        -------
        extracted_features: pd.DataFrame
        """
        # add engineered featuures
        df = helper.derived_feats(preprocessed_data)
        # convert feats to dummy
        df = pd.get_dummies(df, dummy_na=True)  # treat na as a separate category
        # remove any col that is all 0s
        df = df.loc[:, (df != 0).any(axis=0)]

        # narrow to good keys
        feat_names = [k for k in df.keys()  # features to use
                      if not 'iai' in k.lower()]
        PECARN_FEAT_NAMES = ['AbdDistention', 'AbdTenderDegree', 'AbdTrauma', 'AbdTrauma_or_SeatBeltSign',
                             'AbdomenPain', 'Costal', 'DecrBreathSound', 'DistractingPain',
                             'FemurFracture', 'GCSScore', 'Hypotension', 'LtCostalTender',
                             'MOI', 'RtCostalTender', 'SeatBeltSign', 'ThoracicTender',
                             'ThoracicTrauma', 'VomitWretch', 'Age', 'Sex'] + \
                            ['Race', 'InitHeartRate', 'InitSysBPRange']  # new ones to consider
        pecarn_feats = set()
        for pecarn_feat in PECARN_FEAT_NAMES:
            for feat_name in feat_names:
                if pecarn_feat in feat_name:
                    pecarn_feats.add(feat_name)
        pecarn_feats = sorted(list(pecarn_feats)) + ['outcome']
        return df[pecarn_feats]

    def split_data(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """Split into 3 sets: training, tuning, testing
        Keep in mind any natural splits (e.g. hospitals)
        Ensure that there are positive points in all splits.

        Parameters
        ----------
        preprocessed_data

        Returns
        -------
        df_train
        df_tune
        df_test
        """
        # 60-20-20 split
        return np.split(preprocessed_data.sample(frac=1, random_state=42),
                        [int(.6 * len(preprocessed_data)),
                         int(.8 * len(preprocessed_data))])

    def get_outcome_name(self) -> str:
        """Should return the name of the outcome we are predicting
        """
        return 'iai_intervention'

    def get_dataset_id(self) -> str:
        """Should return the name of the dataset id
        """
        return 'iai_pecarn'


if __name__ == '__main__':
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True)
    print('shapes', df_train.shape, df_tune.shape, df_test.shape)
