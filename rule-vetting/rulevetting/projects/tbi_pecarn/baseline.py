import numpy as np
import pandas as pd

from rulevetting.templates.model import ModelTemplate

class Baseline(ModelTemplate):
    def __init__(self):
        # query for each rule + resulting predicted probability
        self.rules = [
            ('SFxPalp_1_or_2 == 1', 9.2),
            ('AMS_1 == 1', 3.1),
            ('High_impact_InjSev_3 == 1', 0.9),
            ('LocLen_2_3_4 == 1', 1.3),
            ('HemaLoc_2_or_3 == 1', 0.3),
            ('ActNorm_0 == 1', 0.3),

            # final condition is just something that is always true
            ('ActNorm_0 >= 0', 0.02),
        ]

        self.not_prob = [0.88,
                         0.58,
                         0.28,
                         0.11,
                         0.08,
                         0.04,
                         0.02]

    def predict(self, df_features: pd.DataFrame):
        predicted_probabilities = pd.DataFrame(self.predict_proba(df_features))[1]
        return (predicted_probabilities > 0.005).astype(int)


    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.zeros(n)
        for i in range(n):
            x = X.iloc[[i]]
            for j, rule in enumerate(self.rules):
                query, prob = rule
                if j == len(self.rules) - 1:
                    probs[i] = prob
                elif not x.query(query).empty:
                    probs[i] = prob
                    break
        return np.vstack(((100 - probs)/100, probs/100)).transpose()  # probs (n, 2)



    def print_model(self, df_features: pd.DataFrame):
        predicted_probabilities = pd.DataFrame(self.predict_proba(df_features))[1]
        #df_features['pred'] = predicted_probabilities
        
        str_print = f''
        for j, rule in enumerate(self.rules):
            query, prob = rule
            print("   Remaining patients have ", self.not_prob[j], "% chance of ciTBI")
            if j < len(self.rules)-1:
                print("if ", query, " then ", prob, "% chance of ciTBI")
        str_print += f'test'
        return str_print
