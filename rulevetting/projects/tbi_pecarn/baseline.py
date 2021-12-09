import numpy as np
import pandas as pd

from rulevetting.templates.model import ModelTemplate

class Baseline(ModelTemplate):
    def __init__(self, age_group='old'):
        # query for each rule + resulting predicted probability in young tree
        if age_group == 'young':
            self.rules = [
                #('GCSScore == GCSScore', 0.9),
                ('AMS == 1', 4.13),
                ('HemaBinary == 1', 1.95),
                ('LocBinary == 1', 1.98),
                ('MechBinary == 1', 0.46),
                ('SFxPalpBinary == 1', 4.81),
                ('ActNorm == 1', 0.44),
                ('ActNorm == 0', 0.02)
            ]

        # query for each rule + resulting predicted probability in old tree
        if age_group == 'old':
            self.rules = [
                #('GCSScore == GCSScore', 0.88),
                ('AMS == 1', 4.07),
                ('LocBinary == 1', 1.17),
                ('Vomit == 1', 0.93),
                ('MechBinary == 1', 0.54),
                ('SFxBas == 1', 8.99),
                ('HABinary == 1', 0.94),
                ('HABinary == 0', 7/14663)
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

    def add_var(self, data) :
        df = data.copy()

        df['HemaBinary'] = np.maximum.reduce([df['HemaLoc_Occipital'], df['HemaLoc_Parietal/Temporal']])
        df['LocBinary'] = np.maximum.reduce([df['LocLen_5 sec - 1 min'], df['LocLen_1-5 min'], df['LocLen_>5 min']])
        df['MechBinary'] = df['High_impact_InjSev_High']
        df['HABinary'] = df['HASeverity_Severe']
        df['SeizLen'] = np.maximum.reduce([df['SeizLen_1-5 min'], df['SeizLen_5-15 min'], df['SeizLen_>15 min']])
        df['HemaSizeBinary'] = np.maximum.reduce([df['HemaSize_Large'], df['HemaSize_Medium']])
        df['LocSeparateBinary'] = np.maximum.reduce([df['LOCSeparate_Suspected'], df['LOCSeparate_Yes']])
        df['SFxPalpBinary'] = np.maximum.reduce([df['SFxPalp_Unclear'], df['SFxPalp_Yes']])

        return df

    def predict(self, df_features: pd.DataFrame):
        df = self.add_var(df_features)
        predicted_probabilities = self._traverse_rule(df)
        print(predicted_probabilities)
        return (predicted_probabilities > 0.11).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        df = self.add_var(df_features)
        predicted_probabilities = self._traverse_rule(df) / 100
        return np.vstack((1 - predicted_probabilities, predicted_probabilities)).transpose()

    def print_model(self, df_features):
        df = self.add_var(df_features)
        self._traverse_rule(df)
        return self.str_print