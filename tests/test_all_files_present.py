from os.path import join as oj

import importlib
import os
import unittest

import mrules
import mrules.api.util

DATA_PATH = oj(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


class TestAllFilesPresent(unittest.TestCase):
    def test_all_files_present(self):
        """Check that all required files are present
        """
        print('testing')
        for project_id in mrules.api.util.get_project_ids():
            project_dir = oj(mrules.PROJECTS_PATH, project_id)
            project_files = os.listdir(project_dir)
            assert 'readme.md' in project_files
            assert 'data_dictionary.md' in project_files
            assert '__init__.py' in project_files
            assert 'dataset.py' in project_files
            assert 'baseline.py' in project_files