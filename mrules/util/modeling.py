'''
# Modeling

- run many models from [imodels](https://github.com/csinva/imodels)
- extract out stable rules: screen for high predictive acc, look at what is kept
- build stable rules model (e.g. using RuleFit or Corels)
'''

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from mrules.projects.iai_pecarn.dataset import Dataset


def fit_models(train_data: pd.DataFrame, tune_data: pd.DataFrame):
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(tune_data)
    predictor = TabularPredictor(label='outcome')
    predictor.fit(train_data, presets='interpretable', verbosity=2, time_limit=30)
    print(predictor.interpretable_models_summary())
    return predictor


if __name__ == '__main__':
    df_train, df_tune, df_test = Dataset().get_data()
    predictor = fit_models(df_train, df_tune)
