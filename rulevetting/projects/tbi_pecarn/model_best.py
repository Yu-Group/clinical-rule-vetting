import numpy as np
import pandas as pd
import pickle as pkl
from os.path import join as oj
from rulevetting.templates.model import ModelTemplate


class Model(ModelTemplate):
    def __init__(self):
        # read the rulefit model
        self.rulefit = pkl.load(open(oj('./rulevetting/projects/tbi_pecarn/model_best.pkl'), 'rb'))
        # get the rules
        self.rules = self.rulefit.get_rules()


    def predict(self, df_features: pd.DataFrame):
        # Rulefit model need to specify X(features)
        df = df_features.drop(columns=['AgeinYears', 'Race', 'Gender']) #metakeys removed
        X = df.drop(columns="outcome")
        # use the predict_prob function in rulefit model
        predicted_probabilities = self.rulefit.predict_proba(X)[:,1]
        return (predicted_probabilities >  1.3e-05).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        # Rulefit model need to specify X(features)
        df = df_features.drop(columns=['AgeinYears', 'Race', 'Gender']) #metakeys removed
        X = df.drop(columns="outcome")
        # use the predict_prob function in rulefit model
        return self.rulefit.predict_proba(X)

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

    # default judgment call
    df_train, df_tune, df_test = Dataset().get_data(split_age=AgeSplit.AGEINVARIANT,load_csvs=False)
    df_full = pd.concat((df_train, df_tune, df_test))
    
    # run the model
    model = Model()
    preds = model.predict(df_full)
    print(model.print_model())
