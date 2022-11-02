import os.path

import numpy as np
import pandas as pd
from imodels.rule_set.rule_fit import RuleFitClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_curve
# from sklearn.metrics import roc_curve, auc
from .tbi_pecarn.dataset import Dataset as tbiDataset
from .csi_pecarn.dataset1 import Dataset as csiDataset
from rulevetting.api.modeling import fit_models


def _get_x_y(ds, columns):
    y_label = "outcome"
    y = ds[y_label]
    X = ds.loc[:, ds.columns.drop(y_label)]
    X = X.loc[:, columns]
    return X, y

def get_specificity(y_true, y_prob_pred, tpr_min):
    roc = roc_curve(y_true=y_true, y_score=y_prob_pred)
    tpr = roc[1]
    fpr = roc[0]

    indx = np.where(tpr >= tpr_min)[0][0]
    return 1-fpr[indx]


if __name__ == '__main__':
    if not os.path.exists("/accounts/campus/omer_ronen/projects/rule-vetting/data/tbi_pecarn/processed/perturbed_data"):
        tbiDataset().get_data(run_perturbations=True, load_csvs=False, save_csvs=True)
    data = tbiDataset().get_data(run_perturbations=True, load_csvs=True, save_csvs=True)
    columns = list(set.intersection(*[set(d[0].columns) for d in list(data.values())]))
    columns.remove("outcome")
    models = []
    for perturb, (train, tune, test) in data.items():
        mdl = RuleFitClassifier()
        X_train, y_train = _get_x_y(train, columns)
        if len(y_train) == 0:
            continue

        mdl.fit(X_train, y_train)
        models.append((perturb, mdl))
    i = 0
    models_spec = {perturb: [] for perturb in data.keys()}
    for perturb, (train, tune, test) in data.items():
        X_test, y_test = _get_x_y(test, columns)
        for p, mdl in enumerate(models):
            y_pred = mdl.predict_proba(X_test)[:, 1]
            spec = get_specificity(y_test, y_pred, tpr_min=0.98)
            models_spec[p].append(spec)
    print(models_spec)
    pd.DataFrame(models_spec).to_csv(f"/accounts/campus/omer_ronen/projects/rule-vetting/results/tbi_98.csv")
