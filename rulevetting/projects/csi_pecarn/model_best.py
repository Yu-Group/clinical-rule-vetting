import numpy as np
import pandas as pd


class SpecialTree:
    def __init__(self):

        self.v_lists = [

        # >12
        ['FocalNeuroFindings', 'HighriskDiving', 'GCSbelowThreshold','PainNeck',
        'SubInj_Head', 'axialloadtop', 'TenderNeck', #'HEENT'
        'HighriskMVC'], 

        # 5-12
        ['FocalNeuroFindings', 'GCSbelowThreshold', 'Torticollis', #'HighriskDiving','HighriskHitByCar'
        'PainNeck', 'Predisposed', 'Clotheslining', 'SubInj_TorsoTrunk', 
        'AxialLoadAnyDoc','HighriskFall','HEENT'], 

        # 2-5
        ['AlteredMentalStatus', 'FocalNeuroFindings', 'Torticollis', 'Predisposed',
        'HighriskMVC', 'PosMidNeckTenderness'],

        # <2
        ['AlteredMentalStatus', 'PosMidNeckTenderness', 'Predisposed', 'AxialLoadAnyDoc',
        'EMSArrival']
        ]

    def predict(self, data):

        for one_list in self.v_lists:
            for v in one_list:
                if v not in data.columns:
                    print('cannot find ' + v)
                    return

        pred = []
        n = data.shape[0]

        for i in range(n):
            df = data.iloc[i]

            if df['VeryYoung'] == 1:
                v_list = self.v_lists[3]
            elif (df['NonVerbal'] == 1) & (df['VeryYoung'] == 0):
                v_list = self.v_lists[2]
            elif (df['YoungAdult'] == 0) & (df['NonVerbal'] == 0):
                v_list = self.v_lists[1]
            else:
                v_list = self.v_lists[0]

            indicator = df[v_list].sum().astype('int')

            if indicator > 0:
                pred.append(1)
            else:
                pred.append(0)

        return np.array(pred)
    
    def print_model(self, df_features: pd.DataFrame):
        '''
        Return string of the model, which includes the number of patients falling into each subgroup.
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
        '''
        
        return ""