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


def get_sensitivity_thres(y_true, y_prob_pred, tpr_min):
    roc = roc_curve(y_true=y_true, y_score=y_prob_pred)

    # fpr = roc[0]
    tpr = roc[1]
    thres = roc[2]

    indx = np.where(tpr >= tpr_min)[0][0]
    return thres[indx]


if __name__ == '__main__':
    tpr = 90
    max_rules = 10

    if not os.path.exists("data/tbi_pecarn/processed/perturbed_data"):
        tbiDataset().get_data(run_perturbations=True, load_csvs=False, save_csvs=True)
    data = tbiDataset().get_data(run_perturbations=True, load_csvs=True, save_csvs=True)
    columns = list(set.intersection(*[set(d[0].columns) for d in list(data.values())]))
    columns.remove("outcome")
    models = []


    def _get_perturb_name(perturb):
        s = perturb.find("[") + 1
        e = perturb.find("]")
        return perturb[s:e].replace("'", "")


    for perturb, (train, tune, test) in data.items():
        print(_get_perturb_name(perturb))
        print(f"Shapes: train - {train.shape[0]}, tune - {tune.shape[0]}, test - {test.shape[0]}")
        _, y_test = _get_x_y(test, columns)
        print(f"test hist: 1: {np.sum(y_test == 1)}, 0: {np.sum(y_test == 0)}")
        mdl = RuleFitClassifier(max_rules=max_rules)
        X_train, y_train = _get_x_y(train, columns)
        X_tune, y_tune = _get_x_y(tune, columns)

        #
        mdl.fit(X_train, y_train)

        description = mdl.visualize(decimals=5)
        description.to_csv(f"results/mdl_{_get_perturb_name(perturb)}_rules_{max_rules}.csv")
        thres = get_sensitivity_thres(y_tune, mdl.predict_proba(X_tune)[:, 1], tpr_min=0.01 * tpr)

        models.append((perturb, mdl, thres))
    # i = 0

    models_spec = {_get_perturb_name(perturb): [] for perturb in data.keys()}
    models_sens = {_get_perturb_name(perturb): [] for perturb in data.keys()}

    for perturb, (train, tune, test) in data.items():
        X_test, y_test = _get_x_y(test, columns)
        for p, mdl, thres in models:
            y_pred = np.array(mdl.predict_proba(X_test)[:, 1] > thres, dtype=np.int)
            sensitivity = np.sum(y_test[y_pred == 1]) / np.sum(y_test)
            specificty = np.sum((1 - y_test)[y_pred == 0]) / np.sum((1 - y_test))
            # spec = get_specificity(y_test, y_pred, tpr_min=0.01*tpr)
            models_spec[_get_perturb_name(p)].append(specificty)
            models_sens[_get_perturb_name(p)].append(sensitivity)

    # print(models_spec)
    pd.DataFrame(models_spec).to_csv(f"results/tbi_spec_{tpr}_rules_{max_rules}.csv")
    pd.DataFrame(models_sens).to_csv(f"results/tbi_sens_{tpr}_rules_{max_rules}.csv")
