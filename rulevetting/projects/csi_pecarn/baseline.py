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
    
    def print_model(self, data):

        pred = self.predict(data)

        n1 = sum(pred)
        n0 = len(pred) - n1

        s = 'Classification summary: '+str(n1) +' patients labeled 1 and ' +str(n0) + ' patients labeled 0.'

        return s

