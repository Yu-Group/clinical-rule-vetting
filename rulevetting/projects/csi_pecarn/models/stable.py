import pickle as pkl
from collections import Counter
from typing import List
from os.path import join as oj

import pandas as pd
import numpy as np
from imodels.rule_set.rule_fit import RuleFit
from imodels.util.score import score_linear
from imodels.util.rule import Rule
from imodels.experimental.util import extract_ensemble, split
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator


def get_max_metric_under_complexity(df: pd.DataFrame,
                                    metric: str,
                                    suffix: str,
                                    c: int) -> float:
    df_under_c = df[df[f'complexity{suffix}'] < c]
    if df_under_c.shape[0] == 0:
        return 0
    max_metric = df_under_c[f'{metric}{suffix}'].sort_values()[-1]

    return max_metric


def get_best_model_rules_under_complexity(df: pd.DataFrame,
                                          metric: str,
                                          suffix: str,
                                          c: int) -> list[Rule]:
    df_under_c = df[df[f'complexity{suffix}'] < c]
    if df_under_c.shape[0] == 0:
        return []

    best_model_idx = df_under_c[f'{metric}{suffix}'].argsort()[-1]
    best_model_vars = df_under_c.iloc[best_model_idx][f'vars{suffix}']
    best_model_rules = best_model_vars['rules_without_feature_names_']
    return best_model_rules


class StableLinear(RuleFit):

    def __init__(self,
                 submodels: List[str] = ['rulefit', 'skope_rules', 'brs'],
                 max_complexity: int = None,
                #  max_complexity_skope_rules: int = None,
                #  max_complexity_brs: int = None,
                 metric: str = None,
                 p_filtering: float = None,
                 min_mult: int = 2,
                 penalty='l1',
                 n_estimators=100,
                 tree_size=4,
                 sample_fract='default',
                 max_rules=30,
                 memory_par=0.01,
                 tree_generator=None,
                 lin_trim_quantile=0.025,
                 lin_standardise=True,
                 exp_rand_tree_size=True,
                 include_linear=False,
                 alpha=None,
                 cv=True,
                 random_state=None):
        super().__init__(n_estimators,
                         tree_size,
                         sample_fract,
                         max_rules,
                         memory_par,
                         tree_generator,
                         lin_trim_quantile,
                         lin_standardise,
                         exp_rand_tree_size,
                         include_linear,
                         alpha,
                         cv,
                         random_state)
        self.max_complexity = max_complexity
        # self.max_complexity_rulefit = max_complexity_rulefit
        # self.max_complexity_skope_rules = max_complexity_skope_rules
        # self.max_complexity_brs = max_complexity_brs
        self.metric = metric
        self.p_filtering = p_filtering
        self.submodels = submodels
        self.penalty = penalty
        self.min_mult = min_mult

    def set_rules(self, dfs: list[pd.DataFrame], suffix: str):
        submodel_metrics = []
        for i, submodel in enumerate(self.submodels):
            submodel_metric = get_max_metric_under_complexity(
                df=dfs[i],
                metric=self.metric,
                suffix=suffix,
                # c=getattr(self, f'max_complexity_{submodel}')
                c=self.max_complexity)
            submodel_metrics.append(submodel_metric)

        all_rules = []
        all_subterms = []
        for i, submodel in enumerate(self.submodels):
            
            if self.p_filtering is not None:
                metric_inclusion_lower_bound = max(submodel_metrics) * (1 - self.p_filtering)
            else:
                metric_inclusion_lower_bound = 0

            if submodel_metrics[i] >= metric_inclusion_lower_bound:
                submodel_rules = get_best_model_rules_under_complexity(
                    df=dfs[i],
                    metric=self.metric,
                    suffix=suffix,
                    # c=getattr(self, f'max_complexity_{submodel}'))
                    c=self.max_complexity)
                all_rules += submodel_rules
                all_subterms += [indv_r for r in submodel_rules for indv_r in split(r)]

        # match full_rules
        repeated_full_rules_counter = {
            k: v for k, v in Counter(all_rules).items() if v >= self.min_mult}
        repeated_rules = set(repeated_full_rules_counter.keys())

        # match subterms of rules
        repeated_subterm_counter = {
            k: v for k, v in Counter(all_subterms).items() if v >= self.min_mult}
        repeated_rules = repeated_rules.union(set(repeated_subterm_counter.keys()))

        # convert to str form to be rescored
        repeated_rules = list(map(str, repeated_rules))
        self.extracted_rules_ = repeated_rules
        return self

    def fit(self, X, y=None, feature_names=None):
        super().fit(X, y, feature_names=feature_names)
        return self

    def _extract_rules(self, X, y) -> List[str]:
        if hasattr(self, 'extracted_rules_'):
            return self.extracted_rules_
        else:
            return extract_ensemble(self.weak_learners, X, y, self.min_mult)

class StableLinearRegressor(StableLinear, RegressorMixin):
    def _init_prediction_task(self):
        self.prediction_task = 'regression'


class StableLinearClassifier(StableLinear, ClassifierMixin):
    def _init_prediction_task(self):
        self.prediction_task = 'classification'
