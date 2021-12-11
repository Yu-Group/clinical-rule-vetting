## our basline model is in "notebooks/baseline.ipynb" ##

import numpy as np
import pandas as pd

from rulevetting.templates.model import ModelTemplate

class Baseline(ModelTemplate):
    def __init__(self):
        pass

    def _traverse_rule(self, df_features: pd.DataFrame):
        str_print = f''
        predicted_probabilities = pd.Series(index=df_features.index, dtype=float).values
        predicted_probabilities = np.ones_like(predicted_probabilities)
        self.str_print = str_print
        print('our basline model is in "notebooks/baseline.ipynb", not here')
        return predicted_probabilities

    def predict(self, df_features: pd.DataFrame):
        predicted_probabilities = self._traverse_rule(df_features)
        return (predicted_probabilities > 0.11).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        predicted_probabilities = self._traverse_rule(df_features) / 100
        return np.vstack((1 - predicted_probabilities, predicted_probabilities)).transpose()

    def print_model(self, df_features):     
        self._traverse_rule(df_features)
        return self.str_print