"""
# Modeling

- run many models from [imodels](https://github.com/csinva/imodels)
- extract out stable rules: screen for high predictive acc, look at what is kept
- build stable rules model (e.g. using RuleFit or Corels)
"""

import numpy as np
import pandas as pd
import random
from autogluon.tabular import TabularDataset, TabularPredictor

import mrules
from mrules.projects.iai_pecarn.dataset import Dataset


def fit_models(train_data: pd.DataFrame, tune_data: pd.DataFrame, interpretable: bool = True):
    """

    Parameters
    ----------
    train_data
    tune_data
    interpretable: bool
        Whether to fit interpretable models or standard models

    Returns
    -------
    predictor
    """
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(tune_data)
    predictor = TabularPredictor(label='outcome', path=mrules.AUTOGLUON_CACHE_PATH)
    kwargs = dict(
        verbosity=2,
        time_limit=30,
    )
    if interpretable:
        predictor.fit(train_data, presets='interpretable', **kwargs)
        print(predictor.interpretable_models_summary())
    else:
        predictor.fit(train_data, **kwargs)
    return predictor


if __name__ == '__main__':
    # todo: loop over datasets (see test_datasets.py)
    np.random.seed(0)
    random.seed(0)
    df_train, df_tune, df_test = Dataset().get_data()
    predictor = fit_models(df_train, df_tune)
    print(predictor)
