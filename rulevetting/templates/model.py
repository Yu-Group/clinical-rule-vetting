import pandas as pd
from abc import abstractmethod


class ModelTemplate:
    """Class for implementing model similar to sklearn model (but without fit method).
    Classes that use this template should be called "Model" for a new model or "Baseline" for a reference model.
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

    def print_model(self, df_features: pd.DataFrame):
        """Return string of the model, which includes the number of patients falling into each subgroup.
        Note this should be the same as the hardcoded values used in the predict function.
        If the model is the baseline used in a paper, it should match it as closely as possible.

        Params
        ------
        df_features: pd.DataFrame
            Path to all data files

        Returns
        -------
        s: str
            Printed version of the existing rule (with number of patients falling into each subgroup).
        """
        return NotImplemented