from os.path import join as oj

import importlib
import numpy as np
import os

import rulevetting
import rulevetting.api.util

DATA_PATH = oj(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def test_models(project):
    """Check that each baseline is implemented properly
    """
    if not project == 'None':
        project_ids = [project]
    else:
        project_ids = rulevetting.api.util.get_project_ids()

    for project_id in project_ids:
        # get data
        project_dset_module_name = f'rulevetting.projects.{project_id}.dataset'
        dset = importlib.import_module(project_dset_module_name).Dataset()
        _, df_tune, _ = dset.get_data(data_path=DATA_PATH, load_csvs=True)
        assert df_tune.shape[0], 'df_tune should not be empty when loading from csvs'

        project_baseline_module_name = f'rulevetting.projects.{project_id}.baseline'
        baseline = importlib.import_module(project_baseline_module_name).Baseline()
        preds_proba = baseline.predict_proba(df_tune)
        assert len(preds_proba.shape) == 2
        assert preds_proba.shape[1] == 2
        assert preds_proba.shape[0] == df_tune.shape[0]
        assert np.max(preds_proba) <= 1, 'predicted probabilities must be <= 1'
        assert np.min(preds_proba) >= 0, 'predicted probabilities must be >= 0'

        preds = baseline.predict(df_tune)
        assert np.array_equal(preds, preds.astype(bool)), 'preds values must only be 0 or 1!'
        assert preds.shape[0] == df_tune.shape[0]

        s = baseline.print_model(df_tune)
        assert isinstance(s, str)
