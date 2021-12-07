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
            "Sensitivity" : sens, "Specificity" : spec, "NPV" : npv}


def fit_eval_lr(X_train, y_train, X_tune, y_tune, title_str, lamb_vec = [0.05, 0.1, 0.2, 0.4, 0.7, 1, 1.3]):

    '''
    Performs (L2-regularized) logistic regression 
    lamb_vec : lambdas (regularization hyperparameter) to tune
    '''
    
    # Now fitting logistic regression
    logreg_model = LogisticRegression(solver='liblinear', random_state = 0).fit(X_train, y_train)
    roc_auc_score(y_train, logreg_model.predict_proba(X_train)[:, 1])  # Train AUC : 0.9513

    # Tuning logistic regression
    roc_tune = []
    acc_tune = []

    for lamb in lamb_vec :
        logreg_model = LogisticRegression(solver='liblinear', random_state = 0, C = lamb).fit(X_train, y_train)
        roc_tune.append(roc_auc_score(y_tune, logreg_model.predict_proba(X_tune)[:, 1]))
        acc_tune.append(logreg_model.score(X_tune, y_tune))
        
    best_lamb = lamb_vec[np.argmax(np.array(roc_tune))]
      
    # With the best lambda
    logreg_model = LogisticRegression(solver='liblinear', penalty="l1", random_state = 0, C = best_lamb).fit(X_train, y_train)

    # statistics update
    stats = predict_stats(logreg_model, X_tune, y_tune)
    stats['lambda'] = best_lamb
    stats['model_type'] = 'Logistic Regression'
    
    plt.title(title_str)
    plt.show()
    
    return (logreg_model, stats)

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

