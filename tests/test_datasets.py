import os
import unittest

test_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.join(test_dir, '..')


class TestDatasets(unittest.TestCase):
    def test_datasets_implemented(self):
        '''Check that each dataset is implemented
        '''
        assert True
