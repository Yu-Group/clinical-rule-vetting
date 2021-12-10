import numpy as np
import pandas as pd


from rulevetting.templates.model import ModelTemplate


class Model(ModelTemplate):
    def __init__(self):
        # read the rulefit model
        self.rulefit = pkl.load(open(oj('./rulevetting/projects/tbi_pecarn/model_best.pkl'), 'rb'))
        # get the rules
        self.rules = self.rulefit.get_rules()


    def predict(self, df_features: pd.DataFrame):
        predicted_probabilities = self.predict_proba(df_features)[:,1]
        return (predicted_probabilities >  1.3e-05).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        # use the predict_prob function in rulefit model
        return self.rulefit.predict_proba(df_features)

    def print_model(self):
        # get rules with non-zero coefficients and print them in the order of importance
        rules = self.rules
        rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
        return rules


if __name__ == '__main__':
    from rulevetting.projects.tbi_pecarn.dataset import Dataset
    from rulevetting.projects.tbi_pecarn.dataset import AgeSplit
    import pickle as pkl
    from os.path import join as oj


    df_train, df_tune, df_test = Dataset().get_data(split_age=AgeSplit.AGEINVARIANT,load_csvs=False)
    df_full = pd.concat((df_train, df_tune, df_test))
    df_full = df_full.drop(columns=['AgeinYears', 'Race', 'Gender'])
    X_full = df_full.drop(columns="outcome")
    y_full = df_full["outcome"].values

    model = Model()
    preds = model.predict(X_full)
    print(model.print_model())

