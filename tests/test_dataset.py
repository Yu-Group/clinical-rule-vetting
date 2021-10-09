from os.path import join as oj

import importlib
import os
import unittest

import mrules
from os.path import join as oj
DATA_PATH = oj(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

class TestDatasets(unittest.TestCase):
    def test_datasets_implemented(self):
        '''Check that each dataset is implemented
        '''
        project_ids = [
            f for f in os.listdir(mrules.PROJECTS_PATH)
            if os.path.isdir(oj(mrules.PROJECTS_PATH, f))
               and not 'cache' in f
        ]
        # print(project_ids, os.listdir(mrules.PROJECTS_PATH))

        project_module_names = [f'mrules.projects.{project_id}.dataset'
                                for project_id in project_ids]
        for project_module_name in project_module_names:

            module = importlib.import_module(project_module_name)
            dset = module.Dataset()

            # pipeline should run
            df_train, df_tune, df_test = dset.get_data(data_path=DATA_PATH, save_csvs=False)
            assert df_tune.shape[0] > 0, 'Tune set must not be empty'
            assert df_train.shape[0] > df_tune.shape[0], 'Train set should be larger than tune set'

            # strings should be filled in
            str_funcs = [dset.get_outcome_name, dset.get_dataset_id]
            for str_func in str_funcs:
                s = str_func()
                assert isinstance(s, str), f'Metadata function {s} should return a string'
