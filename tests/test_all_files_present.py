from os.path import join as oj

import os

import rulevetting
import rulevetting.api.util

PROJECT_PATH = oj(os.path.dirname(os.path.abspath(__file__)), '..', 'rulevetting', 'projects')


def test_all_files_present(project):
    """Check that all required files are present
    """
    if not project == 'None':
        project_ids = [project]
    else:
        project_ids = rulevetting.api.util.get_project_ids()
    for project_id in project_ids:
        project_dir = oj(PROJECT_PATH, project_id)
        project_files = os.listdir(project_dir)
        assert 'readme.md' in project_files
        assert 'data_dictionary.md' in project_files
        assert '__init__.py' in project_files
        assert 'dataset.py' in project_files
        assert 'model_best.py' in project_files
        assert 'baseline.py' in project_files
