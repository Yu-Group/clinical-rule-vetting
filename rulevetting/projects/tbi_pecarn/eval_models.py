import warnings

import os
import pickle as pkl
from os.path import join as oj
from io import StringIO
from IPython.display import Image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, _tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from logitboost import LogitBoost
# from autogluon.tabular import TabularDataset, TabularPredictor

import imodels
import sys
sys.path.append('../../../../')
from rulevetting.api import validation
from rulevetting.projects.tbi_pecarn.dataset import Dataset
from rulevetting.projects.tbi_pecarn.graph import barplot
from rulevetting.projects.tbi_pecarn.baseline import Baseline

data_path = '../../../../data/' # path to raw csv - change to processed...

# default plotting properties - has to be an easier way then doing this every notebook
TINY_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
MARKER_SIZE = 6
LINE_SIZE = 4

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=TINY_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", markersize=MARKER_SIZE)  # marker size
plt.rc("lines", linewidth=LINE_SIZE)  # line width

mpl.rcParams["figure.dpi"] = 180

# Height and width per row and column of subplots
FIG_HEIGHT = 20
FIG_WIDTH = 18
fig_fcn = lambda kwargs: plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), **kwargs)
color_list = sns.color_palette("colorblind")

def predict_and_save(model, X_train, y_train, X_tune, y_tune, tune_only = False, save = True, model_name = 'decision_tree'):
    '''Plots cv and returns cv, saves all stats
    model     : Statistical model object
    tune_only : Boolean, whether getting stats for tuned data only or not
    save      : Boolean, save the result to pickle file or not
    '''
    results = {'model': model}
    for x, y, suffix in zip([X_train, X_tune],
                            [y_train, y_tune],
                            ['_train', '_tune']):
        if tune_only and suffix == '_train' :
            continue
            
        stats, threshes = validation.all_stats_curve(y, model.predict_proba(x)[:, 1],
                                                     plot=suffix == '_tune')
        for stat in stats.keys():
            results[stat + suffix] = stats[stat]
        results['threshes' + suffix] = threshes
    if save :
        pkl.dump(results, open(oj(MODELS_DIR, model_name + '.pkl'), 'wb'))
    return stats, threshes


def predict_stats(model, X_tune, y_tune,  min_sens = 0.95, verbose = True) :
    '''
    WRAPPER FOR predict_and_save function.
    
    Choose the threshold for probability which satisfies given sensitivity (min_sens),
    then prints AUC, accuracy, balanced accuracy, sensitivity, and specificity, NPV
    
    input
    ------
    model       : Statistical Model
    X_tune      : Data for tuning (validation)
    y_tune      : True outcome values
    min_sens    : minimum sensitivity to achieve
    verbose     : Print the statistics or not
    
    output
    ------
    Returns the dictionary containing statistics
    (AUC, accuracy, balanced accuracy, sensitivity, specificity, NPV)
    '''
  
    from sklearn.metrics import confusion_matrix
    
    y_pred_prob = model.predict_proba(X_tune)[:, 1]
    stats, threshes =  predict_and_save(model, X_tune, y_tune, X_tune, y_tune, tune_only = True, save = False, model_name = '')
    np.array(stats['sens']) > min_sens
    
    for i in range(len(stats['sens'])) :
        if stats['sens'][i] > min_sens : 
            continue
        thresh_val = threshes[max(0, i-1)]
        break
      
    y_tune_pred = np.array(y_pred_prob) > thresh_val

    # Confusion Matrix
    cm = confusion_matrix(y_tune, y_tune_pred, labels=[1, 0])
    
    # Calculating statistics
    n = sum(sum(cm))
    auc = roc_auc_score(y_tune, y_pred_prob)
    acc = (cm[0,0] + cm[1,1]) / n
    sens = cm[0,0] / (cm[0,0] + cm[0,1])
    spec = cm[1,1] / (cm[1,0] + cm[1,1])
    npv = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    balacc = (sens + spec) / 2
    
    if verbose :
        print('Confusion Matrix : \n', cm)
        print(f'Prob. threshold : {thresh_val:.5f}')
        print(f'AUC             : {auc:.3f}')  
        print(f'Accuracy        : {acc:.3f}')
        print(f'Sensitivity     : {sens:.3f}')
        print(f'Specificity     : {spec:.3f}')
        print(f'Balanced Acc.   : {balacc:.3f}')
        print(f'NPV             : {npv:.3f}')

    return {'AUC' : auc, 'Accuracy' : acc, 'Balanced Accuracy' : balacc, 
            "Sensitivity" : sens, "Specificity" : spec, "NPV" : npv, "threshold" : thresh_val}


def fit_eval_lr(X_train, y_train, X_tune, y_tune, title_str, lamb_vec = np.logspace(-1, 4, 20)):

    '''
    Performs (L1-regularized) logistic regression 
    lamb_vec : lambdas (regularization hyperparameter) to tune
    '''
    
    # Now fitting logistic regression
    model = LogisticRegression(solver='liblinear', penalty="l1", random_state = 0).fit(X_train, y_train)
    roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])  # Train AUC : 0.9513

    # Tuning logistic regression
    roc_tune = []
    acc_tune = []

    for lamb in lamb_vec :
        model = LogisticRegression(solver='liblinear', penalty="l1",  random_state = 0, C = lamb).fit(X_train, y_train)
        acc_tune.append(model.score(X_tune, y_tune))
        roc_tune.append(roc_auc_score(y_tune, model.predict_proba(X_tune)[:, 1]))
        
        
    best_lamb = lamb_vec[np.argmax(np.array(roc_tune))]
      
    # With the best lambda
    model = LogisticRegression(solver='liblinear', penalty="l1", random_state = 0, C = best_lamb).fit(X_train, y_train)

    # statistics update
    stats = predict_stats(model, X_tune, y_tune)
    stats['lambda'] = best_lamb
    stats['model_type'] = 'Logistic Regression'
    
    plt.title(title_str)
    plt.show()
    
    return (model, stats)


def fit_eval_grouplr(X_train, y_train, X_tune, y_tune, title_str, lamb_vec = np.logspace(-4, -2, 10)):
    '''
    Performs grouped-LASSO logistic regression
    '''
    
    from groupyr import LogisticSGL
    
    if X_train.shape[0] < 100 :
        # Simple case
        group_cov = None
    
    elif 'AgeTwoPlus' in X_train.columns :
        group_cov = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]),
                  np.array([6,7,8,9,10,11]), np.array([12]), np.array([13,14,15,16,17,18]), np.array([19]), 
                  np.array([20,21,22,23,24,25,26]), np.array([27,28,29,30,31,32]), np.array([33,34,35,36,37,38,39,40]), 
                   np.array([41]), np.array([42]), np.array([43,44,45,46,47,48,49,50,51,52,53,54,55]), np.array([56,57,58]), 
                   np.array([59,60,61]), np.array([62,63,64]), np.array([65,66,67,68,69]), np.array([70,71,72,73]), 
                   np.array([74,75,76,77,78]), np.array([79,80,81]), np.array([82,83,84,85]), np.array([86,87,88,89,90]), 
                   np.array([91,92,93,94]), np.array([95,96,97,98,99]), np.array([100,101,102,103]), 
                   np.array([104,105,106,107,108,109]), np.array([110,111,112,113]), np.array([114,115,116,117])]
    else :
        group_cov = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]),
                  np.array([6,7,8,9,10,11]), np.array([12]), np.array([13,14,15,16,17,18]), np.array([19]), 
                  np.array([20,21,22,23,24,25,26]), np.array([27,28,29,30,31,32]), np.array([33,34,35,36,37,38,39,40]), 
                   np.array([41]), np.array([42,43,44,45,46,47,48,49,50,51,52,53,54]), np.array([55,56,57]), 
                   np.array([58,59,60]), np.array([61,62,63]), np.array([64,65,66,67,68]), np.array([69,70,71,72]), 
                   np.array([73,74,75,76,77]), np.array([78,79,80]), np.array([81,82,83,84]), np.array([85,86,87,88,89]), 
                   np.array([90,91,92,93]), np.array([94,95,96,97,98]), np.array([99,100,101,102]), 
                   np.array([103,104,105,106,107,108]), np.array([109,110,111,112]), np.array([113,114,115,116])]

    # Tuning logistic regression
    fpr_tune = []
    
    for lamb in lamb_vec :
        model = LogisticSGL(l1_ratio = 0, alpha = lamb, groups = group_cov).fit(X_train, y_train)
        
        tpr, fpr, thresh = roc_curve(y_tune, model.predict_proba(X_tune)[:, 1])
        
        fpr_f = fpr[np.min(np.where(tpr >= 0.95))]
        
        fpr_tune.append(fpr_f)
    
    best_lamb = lamb_vec[np.argmax(np.array(fpr_tune))]
      
    # With the best lambda
    model = LogisticSGL(l1_ratio = 0, alpha = best_lamb, groups = group_cov).fit(X_train, y_train)

    # statistics update
    stats = predict_stats(model, X_tune, y_tune)
    stats['lambda'] = best_lamb
    stats['model_type'] = 'Grouped Lasso LR'
    
    print("Coefficient for each variable is as following :")
    for var, coef in zip(X_tune.columns, model.coef_) :
        print(f"{var:15} : {coef:5.5f}")
    
    plt.title(title_str)
    plt.show()
    
    return (model, stats)


def fit_eval_boosted(X_train, y_train, X_tune, y_tune, title_str, which_boost = 'AdaBoost', n_estimator = 100):
    '''
    Performs LogitBoost or AdaBoost
    
    input
    -------
    which_boost : 'AdaBoost' or 'LogitBoost'
    n_estimator : The number of weak learners for boosting algorithms
    
    '''
    
    X_val = X_tune.copy()
    y_val = y_tune.copy()
    
    # Fit model
    if which_boost == 'LogitBoost' :
        model = LogitBoost(n_estimators = n_estimator, random_state = 0)
    elif which_boost == 'AdaBoost' :
        model = AdaBoostClassifier(n_estimators = n_estimator, random_state = 0)
    else : 
        print('ERROR: which_boost should be "LogitBoost" or "AdaBoost"')
        return
    
    model.fit(X_train, y_train)
    
    # Find accuracies on train/val sets
    # This takes ~2 minutes to run
    auc_train = []
    acc_train = []
    auc_val = []
    acc_val = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_pred = list(model.staged_predict_proba(X_train))
        val_pred = list(model.staged_predict_proba(X_val))
        for tp in train_pred:
            auc_train.append(roc_auc_score(y_train, tp[:, 1]))
            acc_train.append((y_train == tp.argmax(axis=1)).mean())
            
        for vp in val_pred:
            auc_val.append(roc_auc_score(y_val, vp[:, 1]))
            acc_val.append((y_val == vp.argmax(axis=1)).mean())

    # Print out the stats
    stats = predict_stats(model, X_val, y_val)
    stats['n_estimator'] = n_estimator
    stats['model_type'] = which_boost
    
    plt.title(title_str)
    plt.show()
            
    # Plot ACC/AUC as function of number of weak learners
    plt.figure()
    plt.suptitle(title_str + " Performance", y=1.02)

    plt.subplot(1, 2, 1)
    plt.plot(acc_train, label="Train ACC", color=color_list[0])
    plt.plot(auc_train, label="Train AUC", color=color_list[1])
    plt.xlabel("Number of Weak Learners")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_val, label="Val ACC", color=color_list[2])
    plt.plot(auc_val, label="Val AUC", color=color_list[3])
    plt.xlabel("Number of Weak Learners")
    plt.legend()

    plt.tight_layout()
    
    # Plot Feature Importances
    barplot(pd.Series(dict(zip(X_train.columns, 
                           model.feature_importances_))).sort_values(ascending=False),
        False, title_str + " Feature Importance (Gini)"
       )
    
    return (model, stats)

def fit_eval_decTree(X_train, y_train, X_tune, y_tune, title_str, depth = list(range(1, 6)), weight_ratio = [1, 100]):
    '''
    Performs Decision Tree Algorithm
    
    input
    -------
    depth  : max_depth for decision trees
    weight_ratio : weight_ratio for non-tbi / tbi  (non-tbi : 1, tbi : weight_ratio)
    '''

    # Tuning the depth of the tree
    roc_tune = []
    acc_tune = []
    dep_iter = []
    wr_iter = []
    
    for dep in depth:
        for wr in weight_ratio :
            dt = DecisionTreeClassifier(max_depth = dep, class_weight = {0 : 1, 1 : wr})
            dt.fit(X_train, y_train)
            
            roc_tune.append(roc_auc_score(y_tune, dt.predict_proba(X_tune)[:, 1]))
            acc_tune.append(dt.score(X_tune, y_tune))
            dep_iter.append(dep)
            wr_iter.append(wr)

    # Find max_depth with best performance
    best_dep = dep_iter[np.argmax(np.array(roc_tune))]
    best_wr = wr_iter[np.argmax(np.array(roc_tune))]
    model = DecisionTreeClassifier(max_depth = best_dep, class_weight = {0 : 1, 1 : best_wr}).fit(X_train, y_train)
     
    # Print out the stats
    stats = predict_stats(model, X_tune, y_tune)
    stats['depth'] = best_dep
    stats['weight_ratio'] = best_wr
    stats['model_type'] = 'Decision Tree'
    
    plt.title(title_str)
    plt.show()
    
    return (model, stats)



def fit_eval_svm(X_train, y_train, X_tune, y_tune, title_str, gamma_vec = [2**(-2), 2**(-1.5), 2**(-1), 2**(-0.5), 1, 2**(1)]):
    '''
    Performs SVM
    
    input
    -------
    gamma  : hyperparameter to tune
    '''

    roc_tune = []
    acc_tune = []
    for c in gamma_vec:
        svm_v = svm.SVC(C = c, probability=True).fit(X_train, y_train)  # may add , class_weight = 'balanced'
        roc_tune.append(roc_auc_score(y_tune, svm_v.predict_proba(X_tune)[:, 1]))
        acc_tune.append(svm_v.score(X_tune, y_tune))

    # Find max_depth with best performance
    best_gamma = gamma_vec[np.argmax(np.array(roc_tune))]
    model = svm.SVC(C = best_gamma, probability=True).fit(X_train, y_train)
     
    # Print out the stats
    stats = predict_stats(model, X_tune, y_tune)
    stats['gamma'] = best_gamma
    stats['model_type'] = 'SVM'

    plt.title(title_str)
    plt.show()
    
    return (model, stats)
    


def fit_eval_rf(X_train, y_train, X_tune, y_tune, title_str):
    '''
    Performs Random Forest

    '''
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Print out the stats
    stats = predict_stats(model, X_tune, y_tune)
    stats['model_type'] = 'Random Forest'
    
    plt.title(title_str)
    plt.show()
    
    return (model, stats)

