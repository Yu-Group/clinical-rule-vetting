import pandas as pd
from abc import abstractmethod


class BaselineTemplate:
    """Class for implementing baseline similar to sklearn model (but without fit method).
    Classes that use this template should be called "Baseline"
    """

    @abstractmethod
    def predict(self, df_features: pd.DataFrame):
        """Return binary predictions, exactly following previous paper implementations.

        Params
        ------
        df_features: pd.DataFrame
            Input features

        Returns
        -------
        predictions: array_like (n, 1)
            Values should all be 0 or 1
        """
        return NotImplemented

    @abstractmethod
    def predict_proba(self, df_features: pd.DataFrame):
        """Return probabilistic predictions

        Params
        ------
        df_features: pd.DataFrame
            Path to all data files

        Returns
        -------
        predicted_probabilities: array_like (n, 2)
            Values should be in [0, 1]
            predicted_probabilities[:, 0] should be for class 0
            predicted_probabilities[:, 1] should be for class 1
        """
        return NotImplemented

    def __str__(self):
        """Print the baseline model, allong with the number of patients falling into each subgroup.
        Note this should be the same as the hardcoded values used in the predict function.
        However, it is possible that they are slightly different, in case the data does not exactly match the original paper.

        Returns
        -------
        s: str
            Printed version of the existing rule (with number of patients falling into each subgroup).
        """
        return NotImplemented
