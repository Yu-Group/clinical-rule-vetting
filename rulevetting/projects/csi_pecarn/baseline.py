import numpy as np
import pandas as pd


class Baseline:
    def __init__(self):
        # query for each rule + resulting predicted probability
        self.v_list = ['AlteredMentalStatus','FocalNeuroFindings','PainNeck',
           'Torticollis','SubInj_TorsoTrunk',
           'Predisposed','HighriskDiving','HighriskMVC']

    def predict(self, data):

        for v in self.v_list:
            if v not in data.columns:
                print('cannot find ' + v)
                return

        indicator = data[self.v_list].sum(axis = 1)
        pred = np.array([1 if (i > 0) else 0 for i in indicator])
        return pred
    
    def predict_proba(self, data):

        for v in self.v_list:
            if v not in data.columns:
                print('cannot find ' + v)
                return

        indicator = data[self.v_list].sum(axis = 1)
        pred_binary = np.array([1 if (i > 0) else 0 for i in indicator])
        
        # refactored from
        # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
        prob_result = np.zeros((pred_binary.size, 2))
        prob_result[np.arange(pred_binary.size),pred_binary] = 1
        
        return prob_result

    def print_model(self, df_features: pd.DataFrame):
        """Return string of the model, which includes the number of patients falling into each subgroup.
        Note this should be the same as the hardcoded values used in the predict function.
        If the model is the baseline used in a paper, it should match it as closely as possible.

        Params
        ------
        df_features: pd.DataFrame
            Path to all data files

        Returns
        -------
        s: str
            Printed version of the existing rule (with number of patients falling into each subgroup).
        """
        return ""