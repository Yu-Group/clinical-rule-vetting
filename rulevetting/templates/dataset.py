from os.path import join as oj

import numpy as np
import os
import pandas as pd
import random
from abc import abstractmethod
from joblib import Memory
from typing import Dict

import rulevetting


class DatasetTemplate:
    """Classes that use this template should be called "Dataset"
    All functions take **kwargs, so you can specify any judgement calls you aren't sure about with a kwarg flag.
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
        return NotImplemented

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
        return NotImplemented

    @abstractmethod
    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extract features from preprocessed data
        All features should be binary


        Parameters
        ----------
        preprocessed_data: pd.DataFrame
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        extracted_features: pd.DataFrame
        """
        return NotImplemented

    @abstractmethod
    def split_data(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Split into 3 sets: training, tuning, testing
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
        return NotImplemented

    @abstractmethod
    def get_outcome_name(self) -> str:
        """Should return the name of the outcome we are predicting
        """
        return NotImplemented

    @abstractmethod
    def get_dataset_id(self) -> str:
        """Should return the name of the dataset id (str)
        """
        return NotImplemented

    @abstractmethod
    def get_meta_keys(self) -> list:
        """Return list of keys which should not be used in fitting but are still useful for analysis
        """
        return NotImplemented

    def get_kwargs(self) -> Dict[str, dict]:
        """Return dictionary of keyword arguments for each functions in the dataset class.
        Each key should be a string with the name of the arg.
        Each value should be a list of values, with the default value coming first.
        """
        return {
            'clean_data': {},
            'preprocess_data': {},
            'extract_features': {},
            'split_data': {},
        }

    def get_data(self, save_csvs: bool = False,
                 data_path: str = rulevetting.DATA_PATH,
                 load_csvs: bool = False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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
        kwargs = self.get_kwargs()
        default_kwargs = {}
        for key in kwargs.keys():
            func_kwargs = kwargs[key]
            default_kwargs[key] = {k: func_kwargs[k][0] # first arg in each list is default
                                   for k in func_kwargs.keys()}

        print('kwargs', default_kwargs)
        cleaned_data = cache(self.clean_data)(data_path=data_path, **default_kwargs['clean_data'])
        preprocessed_data = cache(self.preprocess_data)(cleaned_data, **default_kwargs['preprocess_data'])
        extracted_features = cache(self.extract_features)(preprocessed_data, **default_kwargs['extract_features'])
        df_train, df_tune, df_test = cache(self.split_data)(extracted_features, **default_kwargs['split_data'])
        if save_csvs:
            os.makedirs(PROCESSED_PATH, exist_ok=True)
            for df, fname in zip([df_train, df_tune, df_test],
                                 ['train.csv', 'tune.csv', 'test.csv']):
                meta_keys = rulevetting.api.util.get_feat_names_from_base_feats(df.keys(), self.get_meta_keys())
                df.loc[:, meta_keys].to_csv(oj(PROCESSED_PATH, f'meta_{fname}'))
                df.drop(columns=meta_keys).to_csv(oj(PROCESSED_PATH, fname))
        return df_train, df_tune, df_test
