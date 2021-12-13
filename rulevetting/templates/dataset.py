import os
import random
from abc import abstractmethod
from os.path import join as oj
from typing import Dict

import numpy as np
import pandas as pd
from joblib import Memory

import rulevetting
from vflow import init_args, Vset, build_Vset


class DatasetTemplate:
    """Classes that use this template should be called "Dataset"
    All functions take **kwargs, so you can specify any judgement calls you aren't sure about with a kwarg flag.
    Please refrain from shuffling / reordering the data in any of these functions, to ensure a consistent test set.
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

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        """Return dictionary of keyword arguments for each function in the dataset class.
        Each key should be a string with the name of the arg.
        Each value should be a list of values, with the default value coming first.

        Example
        -------
        return {
            'clean_data': {},
            'preprocess_data': {
                'imputation_strategy': ['mean', 'median'],  # first value is default
            },
            'extract_features': {},
        }
        """
        return NotImplemented

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
            extracted_features = cache(self.extract_features)(preprocessed_data, **default_kwargs['extract_features'])
            df_train, df_tune, df_test = cache(self.split_data)(extracted_features)
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

        return df_train, df_tune, df_test
