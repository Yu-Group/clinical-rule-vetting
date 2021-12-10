import math
from typing import List, Dict, Any, Union, Tuple

# import dvu
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import dirname
from util import remove_x_axis_duplicates, merge_overlapping_curves

import os.path
from os.path import join as oj
# dvu.set_style()
mpl.rcParams['figure.dpi'] = 250

cb2 = '#66ccff'
cb = '#1f77b4'
cr = '#cc0000'
cp = '#cc3399'
cy = '#d8b365'
cg = '#5ab4ac'

DIR_FIGS = oj(dirname(os.path.realpath(__file__)), 'figs')


def savefig(fname):
    os.makedirs(DIR_FIGS, exist_ok=True)
    plt.tight_layout()
    # print(oj(DIR_FIGS, fname + '.pdf'))
    plt.savefig(oj(DIR_FIGS, fname + '.pdf'))


def get_x_and_y(result_data: pd.Series, x_col: str, y_col: str, test=False) -> Tuple[np.array, np.array]:
    if test and result_data.index.unique().shape[0] > 1:
        return merge_overlapping_curves(result_data, y_col)

    complexities = result_data[x_col]
    rocs = result_data[y_col]
    complexity_sort_indices = complexities.argsort()
    x = complexities[complexity_sort_indices]
    y = rocs[complexity_sort_indices]
    return remove_x_axis_duplicates(x.values, y.values)


def viz_model_curves_validation(
    ax: plt.Axes,
    result: dict[str, Any],
    suffix: str,
    metric: str = 'rocauc',
    curve_id: str = None) -> None:

    df = result['df']
    if curve_id:
        curve_ids = [curve_id]
    else:
        curve_ids = df['curve_id'].unique()
    dataset = result['dataset']
    
    if suffix == 'test':
        x_column = 'complexity_train'
    else:
        x_column = 'complexity_' + suffix
    y_column = f'{metric}_' + suffix

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    min_complexity_all_curves = float('inf')
    for curve_id in curve_ids:
        curr_curve_df = df[df['curve_id'] == curve_id]
        curr_est_name = curr_curve_df.index[0]

        x, y = get_x_and_y(curr_curve_df, x_column, y_column)
        min_complexity_all_curves = min(min_complexity_all_curves, x[0])

        if len(curve_ids) == 1:
            label = curr_est_name
        else:
            label = curve_id

        ax.plot(x, y, marker='o', markersize=4, label=label)

    ax.set_xlim(0, 30)
    if suffix != 'test':
        est_name_title = curr_est_name
    else:
        est_name_title = 'all'
    ax.set_title(f'{metric} vs. complexity, {est_name_title} on {dataset}')
    ax.set_xlabel('complexity score')
    ax.set_ylabel(y_column)
    ax.legend(frameon=False, handlelength=1)
