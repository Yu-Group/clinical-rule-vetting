from os.path import join as oj

import numpy as np
import os
import pandas as pd
from typing import Dict
import rulevetting.projects.tbi_pecarn.helper as helper

import rulevetting
import rulevetting.api.util
from rulevetting.templates.dataset import DatasetTemplate


class Dataset(DatasetTemplate):
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        '''
        Convert the raw data file into a pandas dataframe.

        Params
        ------
        data_path: str
            Path to all data files
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls
        Returns
        -------
        cleaned_data: pandas DataFrames
        '''
        fname = "TBI PUD 10-08-2013.csv"
        path_to_csv = os.path.join(data_path, "tbi_pecarn", "raw", fname)
        try:
            cleaned_data = pd.read_csv(path_to_csv, encoding="ISO-8859-1")
            return cleaned_data
        except:
            print(f"Error: Raw .csv file is not at {path_to_csv}.")
            return -1

    def preprocess_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        '''
        Preprocesses the dataset by subsetting columns, removing rows, 
        imputing missing data, encoding categorical variables as binary column vectors, 
        and changing the name of the outcome variable. 

        Params
        ------
        data: pd.DataFrame
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls
        Returns
        -------
        preprocessed_data: pd.DataFrame
        '''
        if len(kwargs)==0:
            kwargs = helper.default_judgement_calls_preprocessing()
        print(kwargs)
        # Recalculate outcome variable
        data = helper.recalc_outcome(data)
        
        # Remove patients with missing ciTBI outcomes, GCS below 14,
        # or clear signs they need a CT scan
        data = helper.subset_rows(data, **kwargs)
        
        # Remove columns that are irrelevant, have too many NAs, or 
        # give instant positive diagnoses
        outcome_name = self.get_outcome_name()
        data = helper.subset_cols(data, outcome_name, **kwargs)
        
        # Impute data
        data = helper.impute_data(data, **kwargs)
        
        # Encode categorical variables as binary variables
        data = helper.binarize_data(data, outcome_name, **kwargs)
        
        # Splits dataset by age
        data = helper.split_by_age(data, **kwargs)
        
        # Make new column variables
        data = helper.combine_features(data)

        # Change outcome name to "outcome"
        preprocessed_data = helper.change_outcome_name(data, outcome_name)

        return preprocessed_data

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        '''
        Performs feature selection procedure using logistic regression. 
        Also adds features necessary for baseline model to run.
        
        Params
        ------
        preprocessed_data: pd.DataFrame
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls
        Returns
        -------
        extracted_data: pd.DataFrame
        '''
        # Contains outcome variable and most informative features
        df = helper.derived_feats(preprocessed_data)
        features = list(df.columns)

        # Adds baseline columns
        baseline = self.get_baseline_keys()
        for feature in baseline:
            if feature in preprocessed_data.columns:
                features.append(feature)
        extracted_data = preprocessed_data[features]
        return extracted_data
        
    def get_outcome_name(self) -> str:
        return 'PosIntFinal'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'tbi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        '''
        Return list of keys which should not be used in fitting but are still useful for analysis.
        '''
        return []

    def get_baseline_keys(self) -> list:
        '''
        Returns predictors used in paper's baseline model.
        '''
        return ['AMS_1',
                'AgeTwoPlus_1',
                'HemaLoc_2_or_3',
                'LocLen_2_3_4',
                'High_impact_InjSev_3',
                'SFxPalp_1_or_2',
                'ActNorm_0', 
                'Vomit_1',
                'SFxBas_1',
                'HASeverity_3'
               ]

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        return {
            'clean_data': {},
            'preprocess_data': {
                # drop cols with vals missing over this fraction of the time
                'frac_missing_allowed': [0.1, 0.01],
                
                # Method for imputing missing data
                # KNN works but is slow, so I've commented it out
                'imputation': ["median", "none"],#"KNN",
                
                # Whether or not to exclude patients & variables that clearly indicate a CT scan is necessary
                # Namely, signs of a basilar or palpable skull fracture, or a bulging anterior fontanelle
                'only_mildest_trauma': [False, True],
                
                # "no" = return whole dataset; 
                # "older" = subset children 2 and older; "younger" = age <2
                'split_by_age': ["no", "older", "younger"]
            },
            'extract_features': {},
        }


if __name__ == '__main__':
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=True)
    print('successfuly processed data\nshapes:',
          df_train.shape, df_tune.shape, df_test.shape,
          '\nfeatures:', list(df_train.columns))
