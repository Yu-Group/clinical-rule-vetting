import numpy as np
import pandas as pd
import pickle as pkl
import imodels

from rulevetting.api import validation, util as api_util
from rulevetting.projects.csi_pecarn.dataset import Dataset
from rulevetting.templates.model import ModelTemplate

class Model(ModelTemplate):
    def __init__(self):
        self.loaded_model=pkl.load(open(f'rulefit_model.sav', 'rb'))
        self.thres=0.0019728743590615395

    def _traverse_rule(self, df_features: pd.DataFrame):
        df = df_features.copy()
        o = 'outcome'
        str_print = f''
        false_negative_idx=[]
        prediction=self.predict(df)
        predicted_negative_idx=[]
        for idx, (first,second) in enumerate(zip(prediction,df[o])):
            if first==0 and second==1:
                false_negative_idx.append(idx)
            if first==0:
                predicted_negative_idx.append(idx)
        str_print+=f'{df[o].sum()} / {df.shape[0]} (positive class / total)\n\t\u2193 \n'

        computed_prob_fn = 100 * len(false_negative_idx)/ len(predicted_negative_idx)
        computed_prob_tp = 100 * (df[o].sum()-len(false_negative_idx))/ df[o].sum()
        str_print += f'\033[\033[00m {len(false_negative_idx):>3} / {len(predicted_negative_idx):>4} ({computed_prob_fn:0.1f}%) (false negative / total predicted negative)\n'
        str_print += f'\033[\033[00m {df[o].sum()-len(false_negative_idx):>3} / {df[o].sum():>4} ({computed_prob_tp:0.1f}%) (true positive / total true positive)\n'

        self.str_print = str_print
        return


    def predict(self, df_features: pd.DataFrame):
        predicted_probabilities = self.predict_proba(df_features)
        return (predicted_probabilities[:,1] >= self.thres).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        meta_keys = api_util.get_feat_names_from_base_feats(df_train.columns, Dataset().get_meta_keys())
        df_features=df_features.drop(columns=meta_keys).drop(columns=['outcome'])
        return self.loaded_model.predict_proba(df_features)

    def print_model(self, df_features):
        self._traverse_rule(df_features)
        return self.str_print


if __name__ == '__main__':
    from rulevetting.projects.csi_pecarn.dataset import Dataset

    # df_train, df_tune, df_test = Dataset().get_data(load_csvs=True) # if there are processed data in /data/csi_pecarn/processed
    df_train, df_tune, df_test = Dataset().get_data()

    df_full = pd.concat((df_train, df_tune, df_test))
    model = Model()
    preds_proba = model.predict_proba(df_full)
    print(model.print_model(df_full))
    # preds = baseline.predict(df_train)
    # print('preds_proba', preds_proba.shape, preds_proba[:5])
    # print('preds', preds.shape, preds[:5])
