"""
.. include:: ../readme.md
"""
import os
from os.path import join as oj

MRULES_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = oj(MRULES_PATH, '..')
DATA_PATH = oj(REPO_PATH, 'data')
PROJECTS_PATH = oj(MRULES_PATH, 'projects')
AUTOGLUON_CACHE_PATH = oj(DATA_PATH, 'autogluon_cache')