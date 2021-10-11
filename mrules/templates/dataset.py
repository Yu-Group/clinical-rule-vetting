from os.path import join as oj

import numpy as np
import os
import pandas as pd
import random
from abc import abstractmethod
from joblib import Memory

import mrules


class DatasetTemplate:
    """Classes that use this template should be called "Dataset"
    """

    @abstractmethod
    def clean_data(self, data_path: str = mrules.DATA_PATH) -> pd.DataFrame:
        """
        Convert the raw data files into a pandas dataframe.
        Dataframe keys should be reasonable (lowercase, underscore-separated)

        Params
        ------
        data_path: str, optional
            Path to all data files

        Returns
        -------
        cleaned_data: pd.DataFrame
        """
        return NotImplemented

    @abstractmethod
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
        return NotImplemented

    @abstractmethod
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
        return NotImplemented

    @abstractmethod
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
        return NotImplemented

    @abstractmethod
    def get_outcome_name(self) -> str:
        """Should return the name of the outcome we are predicting
        """
        return NotImplemented

    @abstractmethod
    def get_dataset_id(self) -> str:
        """Should return the name of the dataset id
        """
        return NotImplemented

    @abstractmethod
    def get_meta_keys(self) -> list:
        """Return keys which should not be used in fitting but are still useful for analysis
        """
        return NotImplemented

    def get_data(self, save_csvs: bool = False, data_path: str = mrules.DATA_PATH, load_csvs: bool = False):
        '''Runs all the processing and returns the data.
        This method does not need to be overriden.

        Params
        ------
        save_csvs: bool, optional
            Whether to save csv files of the processed data
        data_path: str, optional
            Path to all data
        load_csvs: bool, optional
            Whether to skip all processing and load data directly from csvs

        Returns
        -------
        df_train
        df_tune
        df_test
        '''
        PROCESSED_PATH = oj(data_path, self.get_dataset_id(), 'processed')
        if load_csvs:
            return tuple([pd.read_csv(oj(PROCESSED_PATH, s), index_col=0)
                          for s in ['train.csv', 'tune.csv', 'test.csv']])
        np.random.seed(0)
        random.seed(0)
        CACHE_PATH = oj(data_path, 'joblib_cache')
        cache = Memory(CACHE_PATH, verbose=0).cache
        cleaned_data = cache(self.clean_data)(data_path=data_path)
        preprocessed_data = cache(self.preprocess_data)(cleaned_data)
        extracted_features = cache(self.extract_features)(preprocessed_data)
        df_train, df_tune, df_test = cache(self.split_data)(extracted_features)
        if save_csvs:
            os.makedirs(PROCESSED_PATH, exist_ok=True)
            for df, fname in zip([df_train, df_tune, df_test],
                                 ['train.csv', 'tune.csv', 'test.csv']):
                meta_keys = mrules.api.util.get_feat_names_from_base_feats(df.keys(), self.get_meta_keys())
                df.loc[:, meta_keys].to_csv(oj(PROCESSED_PATH, f'meta_{fname}'))
                df.drop(columns=meta_keys).to_csv(oj(PROCESSED_PATH, fname))
        return df_train, df_tune, df_test
