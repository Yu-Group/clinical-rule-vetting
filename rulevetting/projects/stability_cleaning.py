import os.path

import numpy as np
import pandas as pd
from imodels import FIGSClassifier, FIGSClassifierCV
from imodels.rule_set.rule_fit import RuleFitClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
    tpr = 95
    max_rules = 30

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


    importances = []

    for perturb, (train, tune, test) in data.items():
        print(_get_perturb_name(perturb))
        print(f"Shapes: train - {train.shape[0]}, tune - {tune.shape[0]}, test - {test.shape[0]}")
        _, y_test = _get_x_y(test, columns)
        print(f"test hist: 1: {np.sum(y_test == 1)}, 0: {np.sum(y_test == 0)}")
        # mdl = FIGSClassifier(max_rules=10)
        mdl = RandomForestClassifier(random_state=0, max_depth=3)
        # mdl = RuleFitClassifier(max_rules=max_rules)
        X_train, y_train = _get_x_y(train, columns)
        X_tune, y_tune = _get_x_y(tune, columns)
        #
        # #
        mdl.fit(X_train, y_train)
        important_mdi = pd.DataFrame({"imp": mdl.feature_importances_, "feature": X_train.columns}).sort_values("imp",
                                                                                                                ascending=False)
        importances.append(important_mdi)
        # important_mdi.to_csv(f"results/rf_{_get_perturb_name(perturb)}_features.csv")
        # mdl.plot()
        # plt.show()
        # description = mdl.visualize(decimals=5)
        # description.to_csv(f"results/mdl_{_get_perturb_name(perturb)}_rules_{max_rules}.csv")

        thres = get_sensitivity_thres(y_tune, mdl.predict_proba(X_tune)[:, 1], tpr_min=0.01 * tpr)
        bs_models = []
        # for i in range(5):
        #     mdl_bs = RuleFitClassifier(max_rules=max_rules)
        #     X_train, y_train = _get_x_y(train.sample(train.shape[0]), columns)
        #     X_tune, y_tune = _get_x_y(tune.sample(tune.shape[0]), columns)
        #     mdl_bs.fit(X_train, y_train)
        #     thres_bs = get_sensitivity_thres(y_tune, mdl_bs.predict_proba(X_tune)[:, 1], tpr_min=0.01 * tpr)
        #
        #     bs_models.append((mdl_bs, thres_bs))

        models.append((perturb, mdl, thres, bs_models))
    imp = pd.concat([i.iloc[:, 0] for i in importances], axis=1, ignore_index=True)
    imp.index = importances[0].iloc[:, 1]
    imp.to_csv("results/rf_features.csv")

    fig, ax = plt.subplots(1)
    ax.matshow(imp, aspect='auto')
    ax.set_ylabel("Feature number")
    ax.set_xlabel("RF model number")

    plt.savefig("results/heatmap.png")
    plt.close()
    # i = 0

    models_spec = {_get_perturb_name(perturb): [] for perturb in data.keys()}
    models_sens = {_get_perturb_name(perturb): [] for perturb in data.keys()}

    models_spec_std = {_get_perturb_name(perturb): [] for perturb in data.keys()}
    models_sens_std = {_get_perturb_name(perturb): [] for perturb in data.keys()}

    for perturb, (train, tune, test) in data.items():
        X_test, y_test = _get_x_y(test, columns)
        for p, mdl, thres, bs_models in models:
            y_pred = np.array(mdl.predict_proba(X_test)[:, 1] > thres, dtype=np.int)
            sensitivity = np.sum(y_test[y_pred == 1]) / np.sum(y_test)
            specificty = np.sum((1 - y_test)[y_pred == 0]) / np.sum((1 - y_test))
            print(f"sens: {sensitivity}, y: {np.mean(y_test)}")
            # spec = get_specificity(y_test, y_pred, tpr_min=0.01*tpr)

            # sens_bs = []
            # spec_bs = []
            # for (bs_mdl, bs_thres) in bs_models:
            #     y_pred = np.array(bs_mdl.predict_proba(X_test)[:, 1] > bs_thres, dtype=np.int)
            #     sensitivity_bs = np.sum(y_test[y_pred == 1]) / np.sum(y_test)
            #     sens_bs.append(sensitivity_bs)
            #     specificty_bs = np.sum((1 - y_test)[y_pred == 0]) / np.sum((1 - y_test))
            #     spec_bs.append(specificty_bs)
            models_spec[_get_perturb_name(p)].append(specificty)
            models_sens[_get_perturb_name(p)].append(sensitivity)
            #
            # models_spec_std[_get_perturb_name(p)].append(np.std(spec_bs))
            # models_sens_std[_get_perturb_name(p)].append(np.std(sens_bs))

    # print(models_spec)
    pd.DataFrame(models_spec).to_csv(f"results/tbi_spec_{tpr}_rules_{max_rules}_rf.csv")
    pd.DataFrame(models_sens).to_csv(f"results/tbi_sens_{tpr}_rules_{max_rules}_rf.csv")

    # pd.DataFrame(models_spec_std).to_csv(f"results/tbi_spec_std_{tpr}_rules_{max_rules}.csv")
    # pd.DataFrame(models_sens_std).to_csv(f"results/tbi_sens_std_{tpr}_rules_{max_rules}.csv")
