from os.path import join as oj

import os

import mrules


def get_project_ids():
    return [
        f for f in os.listdir(mrules.PROJECTS_PATH)
        if os.path.isdir(oj(mrules.PROJECTS_PATH, f))
           and not 'cache' in f
    ]
