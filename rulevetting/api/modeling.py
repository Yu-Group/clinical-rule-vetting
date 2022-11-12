"""
# Modeling

- run many models from [imodels](https://github.com/csinva/imodels)
- extract out stable rules: screen for high predictive acc, look at what is kept
- build stable rules model (e.g. using RuleFit or Corels)
"""

import importlib
import random

import numpy as np
import pandas as pd
# from autogluon.tabular import TabularDataset, TabularPredictor

import rulevetting
import rulevetting.api.util


def fit_models(train_data: pd.DataFrame, tune_data: pd.DataFrame, interpretable: bool = True):
    """Note: don't use this function, still depends on experimental autogluon dependencies

    Parameters
    ----------
    train_datas
    tune_data
    interpretable: bool
        Whether to fit interpretable models or standard models

    Returns
    -------
    predictor
    """
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(tune_data)
    predictor = TabularPredictor(label='outcome',
                                 path=rulevetting.AUTOGLUON_CACHE_PATH,
                                 eval_metric='roc_auc')
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
    project_ids = rulevetting.api.util.get_project_ids()
    for project_id in project_ids:
        np.random.seed(0)
        random.seed(0)
        print('fitting on', project_id)
        project_module_name = f'rulevetting.projects.{project_id}.dataset'
        module = importlib.import_module(project_module_name)
        dset = module.Dataset()
        df_train, df_tune, df_test = dset.get_data(load_csvs=True)
        predictor = fit_models(df_train, df_tune)
        print(predictor)
