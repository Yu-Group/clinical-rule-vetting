import numpy as np
import pandas as pd

from mrules.templates.baseline import BaselineTemplate


class Baseline(BaselineTemplate):
    def __init__(self):
        # query for each rule + resulting predicted probability
        self.rules = [
            ('AbdTrauma_or_SeatBeltSign_yes == 1', 5.7),
            ('GCSScore < 14', 4.6),
            ('AbdTenderDegree_None == 0', 1.4),
            ('ThoracicTrauma_yes == 1', 0.6),
            ('AbdomenPain_yes == 1', 0.7),
            ('DecrBreathSound_yes == 1', 2.9),
            ('VomitWretch_yes == 1', 0.5),

            # final condition is just something that is always true
            ('GCSScore == GCSScore', 0.1),
        ]

    def traverse_rule(self, df_features: pd.DataFrame):
        str_print = f''
        predicted_probabilities = pd.Series(index=df_features.index, dtype=float)
        df = df_features.copy()
        o = 'outcome'
        str_print += f'{"Initial":<35} {df[o].sum()} / {df.shape[0]}\n'
        for rule in self.rules:
            query, prob = rule
            df_rhs = df.query(query)
            idxs_satisfying_rule = df_rhs.index
            predicted_probabilities.loc[idxs_satisfying_rule] = prob

            df.drop(index=idxs_satisfying_rule, inplace=True)
            computed_prob = 100 * df_rhs[o].sum() / df_rhs.shape[0]
            str_print += f'{query:<35} {df[o].sum():>3} / {df.shape[0]:>5}\t {df_rhs[o].sum():>3} / {df_rhs.shape[0]:>4} ({computed_prob:0.1f})\n'
        predicted_probabilities = predicted_probabilities.values
        self.str_print = str_print
        return predicted_probabilities

    def predict(self, df_features: pd.DataFrame):
        predicted_probabilities = self.traverse_rule(df_features)
        return (predicted_probabilities > 0.11).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        predicted_probabilities, _ = self.traverse_rule(df_features)
        return np.vstack((1 - predicted_probabilities, predicted_probabilities)).transpose()

    def __str__(self):
        self.traverse_rule()
        return self.str_print


if __name__ == '__main__':
    from mrules.projects.iai_pecarn.dataset import Dataset

    df_train, df_tune, df_test = Dataset().get_data(load_csvs=True)
    df_full = pd.concat((df_train, df_tune, df_test))
    baseline = Baseline()
    preds_proba = baseline.predict_proba(df_full)
    # preds = baseline.predict(df_train)
    # print('preds_proba', preds_proba.shape, preds_proba[:5])
    # print('preds', preds.shape, preds[:5])
