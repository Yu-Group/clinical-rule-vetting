from os.path import join as oj

import numpy as np
import os
import pandas as pd
import random
from abc import abstractmethod
from joblib import Memory

import mrules


class MDataset:

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

    def get_data(self, save_csvs: bool=False, data_path: str=mrules.DATA_PATH):
        '''Runs all the processing and returns the data

        Params
        ------
        save_csvs: bool, optional
            Whether to save csv files of the processed data
        data_path: str, optional
            Path to all data

        Returns
        -------
        df_train
        df_tune
        df_test
        '''
        np.random.seed(0)
        random.seed(0)
        CACHE_PATH = oj(data_path, 'joblib_cache')
        cache = Memory(CACHE_PATH, verbose=0).cache
        cleaned_data = cache(self.clean_data)(data_path=data_path)
        preprocessed_data = cache(self.preprocess_data)(cleaned_data)
        extracted_features = cache(self.extract_features)(preprocessed_data)
        df_train, df_tune, df_test = cache(self.split_data)(extracted_features)
        if save_csvs:
            PROCESSED_PATH = oj(data_path, self.get_dataset_id(), 'processed')
            os.makedirs(PROCESSED_PATH)
            df_train.to_csv(PROCESSED_PATH, 'train.csv')
            df_train.to_csv(PROCESSED_PATH, 'tune.csv')
            df_train.to_csv(PROCESSED_PATH, 'test.csv')
        return df_train, df_tune, df_test
