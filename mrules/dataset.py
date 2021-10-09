import pandas as pd
from abc import abstractmethod
import random
import numpy as np

class MDataset:

    @abstractmethod
    def clean_data(self) -> pd.DataFrame:
        """
        Convert the raw data files into a pandas dataframe.
        Dataframe keys should be reasonable (lowercase, underscore-separated)

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

    @property
    def get_outcome_name(self) -> str:
        """Should return the name of the outcome we are predicting
        """
        return NotImplemented

    def get_data(self):
        '''Runs all the processing and returns the data

        Returns
        -------
        df_train
        df_tune
        df_test
        '''
        np.random.seed(0)
        random.seed(0)
        cleaned_data = self.clean_data()
        preprocessed_data = self.preprocess_data(cleaned_data)
        extracted_features = self.extract_features(preprocessed_data)
        return self.split_data(extracted_features)
