import warnings
from os.path import join as oj

import sys

sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics


def all_stats_curve(y_test, preds_proba, plot=False, thresholds=None):
    '''preds_proba should be 1d
    '''
    if thresholds is None:
        thresholds = sorted(np.unique(preds_proba))
    all_stats = {
        s: [] for s in ['sens', 'spec', 'ppv', 'npv', 'lr+', 'lr-', 'f1']
    }
    for threshold in tqdm(thresholds):
        preds = preds_proba > threshold
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            all_stats['sens'].append(sens)
            all_stats['spec'].append(spec)
            all_stats['ppv'].append(tp / (tp + fp))
            all_stats['npv'].append(tn / (tn + fn))
            all_stats['lr+'].append(sens / (1 - spec))
            all_stats['lr-'].append((1 - sens) / spec)
            all_stats['f1'].append(tp / (tp + 0.5 * (fp + fn)))

    if plot:
        fig = plt.figure()
        fig = plt.figure(facecolor=(1, 1, 1))
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(t, t ** 2, 'b-')
        ax.plot(all_stats['sens'], all_stats['spec'], '.-')
        ax.grid(True)
        ax.set_ylabel('specificity')
        ax.set_xlabel('sensitivity')
        
        return all_stats, thresholds, fig
        
    else:
        return all_stats, thresholds
