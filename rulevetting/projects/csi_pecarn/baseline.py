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
        pred = [1 if (i > 0) else 0 for i in indicator]
        return pred

