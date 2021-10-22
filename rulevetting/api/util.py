from os.path import join as oj

import os

import rulevetting


def get_project_ids():
    return [
        f for f in os.listdir(rulevetting.PROJECTS_PATH)
        if os.path.isdir(oj(rulevetting.PROJECTS_PATH, f))
           and not 'cache' in f
    ]


def get_feat_names_from_base_feats(feat_names: list, base_feat_names: list):
    """Generate feature names in feat_names that stem from features in base_feats
    """

    feats = set()
    for base_feat_name in base_feat_names:
        for feat_name in feat_names:
            if base_feat_name in feat_name:
                feats.add(feat_name)
    return sorted(list(feats))
