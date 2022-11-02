import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from rulevetting.templates.model import ModelTemplate


class Model(ModelTemplate):
    def __init__(self):
        self.model = load('./notebooks/models/lr_model_all.joblib') 
    
    def predict(self, df_features: pd.DataFrame):
        return self.model.predict(df_features)

    def predict_proba(self, df_features: pd.DataFrame):
        return self.model.predict_proba(df_features)

    def print_model(self, df_features):
        print(self.model)


if __name__ == '__main__':
    from rulevetting.projects.tbi_pecarn.dataset import Dataset

    df_train, df_tune, df_test = Dataset().get_data(load_csvs=True)
    df_full = pd.concat((df_train, df_tune, df_test))
    model = Model()
    preds_proba = model.predict_proba(df_full)
    print(model.print_model(df_full))
    