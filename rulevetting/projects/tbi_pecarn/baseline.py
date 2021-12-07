import numpy as np
import pandas as pd

from rulevetting.templates.model import ModelTemplate

class Baseline(ModelTemplate):
    def __init__(self, age_group):
        # query for each rule + resulting predicted probability in young tree
        if age_group == 'young':
            self.rules = [
                ('GCSScore == GCSScore', 0.9),
                ('AMS == 1', 4.13),
                ('HemaBinary == 1', 1.95),
                ('LocBinary == 1', 1.98),
                ('MechBinary == 1', 0.46),
                ('SFxPalpBinary == 1', 4.81),
                ('ActNorm == 1', 0.44)
            ]
        

        # query for each rule + resulting predicted probability in old tree
        if age_group == 'old':
            self.rules = [
                ('GCSScore == GCSScore', 0.88),
                ('AMS == 1', 4.07),
                ('LocBinary == 1', 1.17),
                ('Vomit == 1', 0.93),
                ('MechBinary == 1', 0.54),
                ('SFxBas == 1', 8.99),
                ('HABinary == 1', 0.94),
            ]

    def _traverse_rule(self, df_features: pd.DataFrame):
        str_print = f''
        predicted_probabilities = pd.Series(index=df_features.index, dtype=float)
        df = df_features.copy()
        o = 'outcome'
        str_print += f'{df[o].sum()} / {df.shape[0]} (positive class / total)\n\t\u2193 \n'
        for j, rule in enumerate(self.rules):
            query, prob = rule
            df_rhs = df.query(query)
            idxs_satisfying_rule = df_rhs.index
            predicted_probabilities.loc[idxs_satisfying_rule] = prob
            df.drop(index=idxs_satisfying_rule, inplace=True)
            computed_prob = 100 * df_rhs[o].sum() / df_rhs.shape[0]
            query_print = query.replace(' == 1', '')
            if j < len(self.rules) - 1:
                str_print += f'\033[96mIf {query_print:<35}\033[00m \u2192 {df_rhs[o].sum():>3} / {df_rhs.shape[0]:>4} ({computed_prob:0.1f}%)\n\t\u2193 \n   {df[o].sum():>3} / {df.shape[0]:>5}\t \n'
        predicted_probabilities = predicted_probabilities.values
        self.str_print = str_print
        return predicted_probabilities

    def predict(self, df_features: pd.DataFrame):
        predicted_probabilities = self._traverse_rule(df_features)
        print(predicted_probabilities)
        return (predicted_probabilities > 0.11).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        predicted_probabilities = self._traverse_rule(df_features) / 100
        return np.vstack((1 - predicted_probabilities, predicted_probabilities)).transpose()

    def print_model(self, df_features):
        self._traverse_rule(df_features)
        return self.str_print