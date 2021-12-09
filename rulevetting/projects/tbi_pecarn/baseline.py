import numpy as np
import pandas as pd

from rulevetting.templates.model import ModelTemplate


class Baseline(ModelTemplate):
    def __init__(self, agegroup: str):
        # query for each rule + resulting predicted probability
        self.agegroup = agegroup
        if self.agegroup == 'young':
            self.rules = [
                ('AMS == 1', 4.1),
                ('HemaLoc == [2, 3]', 1.9),
                ('LocLen == [2, 3, 4]', 2.0),
                ('High_impact_InjSev == 3', 0.5),
                ('SFxPalp == 1', 33.3),
                ('ActNorm == 0', 0.4),

                # final condition is just something that is always true
                ('GCSTotal >= 0', 0.03),
            ]
        if self.agegroup == 'old':
            self.rules = [
                ('AMS == 1', 4.1),
                ('LOCSeparate == [1, 2]', 1.2),
                ('Vomit == 1', 0.9),
                ('High_impact_InjSev == 3', 0.5),
                ('SFxBas == 1', 9.0),
                ('HASeverity == 3', 1.3),

                # final condition is just something that is always true
                ('GCSTotal >= 0', 0.05),
            ]


    def _traverse_rule(self, df_features: pd.DataFrame):
        str_print = f''
        predicted_probabilities = pd.Series(index=df_features.index, dtype=float)
        df = df_features.copy()
        o = 'PosIntFinal'    # outcome variable name
        str_print += f'{df[o].sum()} / {df.shape[0]} (positive class / total)\n\t\u2193 \n'
        for j, rule in enumerate(self.rules):
            query, prob = rule
            df_rhs = df.query(query)
            idxs_satisfying_rule = df_rhs.index
            # the prob we used in rule should be the approx of computed_prob (frequency)
            predicted_probabilities.loc[idxs_satisfying_rule] = prob
            # drop the rows we just assigned prob
            df.drop(index=idxs_satisfying_rule, inplace=True)
            # compute the frequency in percent
            computed_prob = 100 * df_rhs[o].sum() / df_rhs.shape[0]
            query_print = query.replace(' == 1', '') # for print purpose
            if j < len(self.rules) - 1:
                str_print += f'\033[96mIf {query_print:<35}\033[00m \u2192 {df_rhs[o].sum():>3} / {df_rhs.shape[0]:>4} ({computed_prob:0.1f}%)\n\t\u2193 \n   {df[o].sum():>3} / {df.shape[0]:>5}\t \n'
        # we have assigned all patients prob
        predicted_probabilities = predicted_probabilities.values
        self.str_print = str_print
        return predicted_probabilities

    def predict(self, df_features: pd.DataFrame):
        predicted_probabilities = self._traverse_rule(df_features)
        # for each age group, do different prediction
        # (based on the  prob from final condition - the one always true)
        if self.agegroup == "young":
            return (predicted_probabilities > 0.031).astype(int)
        if self.agegroup == "old":
            return (predicted_probabilities > 0.051).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        # convert from percent to value
        predicted_probabilities = self._traverse_rule(df_features) / 100
        return np.vstack((1 - predicted_probabilities, predicted_probabilities)).transpose()

    def print_model(self, df_features):
        self._traverse_rule(df_features)
        return self.str_print

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from rulevetting.projects.tbi_pecarn.dataset import Dataset

    # use original data
    tbi_df = Dataset().clean_data()
    tbi_df.index = tbi_df.PatNum.copy()

    # data processing
    tbi_df = tbi_df[tbi_df['GCSGroup'] == 2]
    tbi_df.drop(tbi_df[tbi_df.PosIntFinal.isnull()].index,
                        inplace=True)



    # divided by ages
    tbi_df_young = tbi_df[tbi_df['AgeinYears'] < 2]
    tbi_df_old = tbi_df[tbi_df['AgeinYears'] >= 2]

    # baseline for age < 2
    model_young = Baseline("young")
    preds_proba = model_young.predict_proba(tbi_df_young)
    print(model_young.print_model(tbi_df_young))

    # baseline for age >= 2
    model_old = Baseline('old')
    print(model_old.print_model(tbi_df_old))