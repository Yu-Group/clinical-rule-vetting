import numpy as np
import pandas as pd

from rulevetting.templates.model import ModelTemplate

class Baseline(ModelTemplate):
    def __init__(self):
        pass

    def _traverse_rule(self, data: pd.DataFrame):
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

    def predict(self, df: pd.DataFrame):
        df = self._traverse_rule(df)
        young_pred = 100 * df['AgeTwoPlus'] + df.loc[:, ['AMS', 'HemaBinary', 'LocBinary', 'MechBinary', 'SFxPalpBinary']].sum(axis=1) + 1 - df['ActNorm']
        old_pred = 100 * df['AgeTwoPlus'] + df.loc[:, ['AMS', 'LocBinary', 'Vomit', 'MechBinary', 'SFxBas', 'HABinary']].sum(axis=1)
        pred = (young_pred != 100.0) & (old_pred != 200.0)

        return pred

    def predict_proba(self, df: pd.DataFrame):
        df = self._traverse_rule(df)
        young_pred = 100 * df['AgeTwoPlus'] + df.loc[:, ['AMS', 'HemaBinary', 'LocBinary', 'MechBinary', 'SFxPalpBinary']].sum(axis=1) + 1 - df['ActNorm']
        old_pred = 100 * df['AgeTwoPlus'] + df.loc[:, ['AMS', 'LocBinary', 'Vomit', 'MechBinary', 'SFxBas', 'HABinary']].sum(axis=1)
        pred = (young_pred != 100.0) & (old_pred != 200.0)
        pred = pred.astype(int)
        pred = np.vstack((pred, 1 - pred)).T
        return pred

    def print_model(self, df_features):
        self._traverse_rule(df_features)
        return str(self._traverse_rule(df_features))