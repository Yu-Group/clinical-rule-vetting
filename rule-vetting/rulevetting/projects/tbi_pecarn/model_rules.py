import numpy as np
import pandas as pd

from rulevetting.templates.model import ModelTemplate


class ModelRules(ModelTemplate):
    def __init__(self):
        # query for each rule + resulting predicted probability
        self.rules = [
            ('AMS_1 == 1', 4.4),
            ('High_impact_InjSev_3 == 1', 1.4),
            ('HemaLoc_3 == 1',0.9 ),
            ('NeuroDOth_1 == 1', 3.1), 
            ('LOCSeparate_2 == 1', 0.9), 
            ('LOCSeparate_1 == 1 ', 0.6), 
            ('VomitNbr_3 == 1', 0.6),
            ('Hema_1 == 1', 0.1),
            

            # final condition is just something that is always true
            ('HAStart_92 == 1', 0),
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
        return (predicted_probabilities > 0.11).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        predicted_probabilities = self._traverse_rule(df_features)
        return np.vstack((1 - predicted_probabilities, predicted_probabilities)).transpose()

    def print_model(self, df_features):
        self._traverse_rule(df_features)
        return self.str_print


if __name__ == '__main__':
    from rulevetting.projects.tbi_pecarn.dataset import Dataset

    df_train, df_tune, df_test = Dataset().get_data(load_csvs=True)
    df_full = pd.concat((df_train, df_tune, df_test))
    model = ModelRules()
    preds_proba = model.predict_proba(df_full)
    print(model.print_model(df_full))