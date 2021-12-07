import os
from os.path import join as oj

import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import imodels

import pickle as pkl
import matplotlib.pyplot as plt

from rulevetting.api import validation_new

'''
Helper functions for fit_models.ipynb.
'''
#---
def var_selection(df,method=['rfe', 10]):
    # Input: a dataframe with outcome as the last column,
    #        method: ['rfe', number of features to choose] or ['lasso',penalty]
    # Output: a dataframe containing the columns we select and the outcome column
    
    algo = method[0]
    param = method[1]
    
    X = df.drop(columns=['outcome'])
    y = df.outcome
    
    if algo == 'rfe':
        mymodel = LogisticRegression()
        myrfe = RFE(mymodel, n_features_to_select = param)
        myfit = myrfe.fit(X, y)
        index = np.append(myfit.support_, True)
    
    elif algo == 'lasso':
        mylasso = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = param)   ## for example C=0.1
        myfit = mylasso.fit(X, y) 
        index = np.append(myfit.coef_[0]!=0, True)
    
    return index

#---
def model_valid(df_train, df_tune, max_num=20, model_name='decision_tree', outcome_def = 'outcome'):
    ## Use validation set to select the number of features to use
    
    record = np.zeros(max_num)
    sensitivity = np.zeros(max_num)
    
    for num in range(1,max_num+1):
        index = var_selection(df_train, method=['rfe', num])
        
        loc_train = df_train.loc[:,index]
        loc_tune = df_tune.loc[:,index]
        #loc_test = df_test.loc[:,index]
        
        X_train = loc_train.drop(columns = outcome_def)
        y_train = loc_train[outcome_def].values
        X_tune = loc_tune.drop(columns = outcome_def)
        y_tune = loc_tune[outcome_def].values
        
        if model_name=='decision_tree':
            model = DecisionTreeClassifier(max_depth=4, class_weight={0: 1, 1: 1e3})
            model.fit(X_train, y_train)
            
        elif model_name=='logistic':
            model= LogisticRegression()
            model.fit(X_train, y_train)
            
        elif model_name=='adaboost':
            model= AdaBoostClassifier(n_estimators=50, learning_rate=1)
            model.fit(X_train, y_train)
        
        stats, threshes = validation_new.all_stats_curve(y_tune, model.predict_proba(X_tune)[:, 1], plot=False)
        
        sens=stats['sens']
        spec=stats['spec']
        
        if sens[0]<0.98:
            record[num-1]=0.
            sensitivity[num-1]=sens[0]
            continue
        
        j=0
        while sens[j]>0.98:
            cur_pec=spec[j]
            j+=1
            
        record[num-1]=cur_pec
        sensitivity[num-1]=sens[j]
    
    print("record:")
    print(record)
    print("sensitivity:")
    print(sensitivity)
    
    return np.argmax(record)+1    ## output the optimal number of features via validation 


#---
def predict_and_save(X_train, X_tune, y_train, y_tune, model, MODELS_DIR, model_name='decision_tree'):
    ## Plots cv and returns cv
    ## Saves all stats

    results = {'model': model}
    for x, y, suffix in zip([X_train, X_tune],
                            [y_train, y_tune],
                            ['_train', '_tune']):
        
        if suffix == '_tune':
            stats, threshes, plot = validation_new.all_stats_curve(y, model.predict_proba(x)[:, 1], plot= True)
        else:
            stats, threshes = validation_new.all_stats_curve(y, model.predict_proba(x)[:, 1], plot= False)
            plot = plt.figure()

        for stat in stats.keys():
            results[stat + suffix] = stats[stat]
        
        results['threshes' + suffix] = threshes
        
    pkl.dump(results, open(oj(MODELS_DIR, model_name + '.pkl'), 'wb'))
    
    return stats, threshes, plot

#---
def fit_simple_models(model_name, X_train, X_tune, y_train, y_tune, feature_names, MODELS_DIR, 
                      max_depth=4, class_weight={0: 1, 1: 1e3}, n_estimators=100, learning_rate=1):

    ## model_name = 'decision_tree', 'logistic', 'adaboost'
    
    if (model_name == 'decision_tree'):
        model = DecisionTreeClassifier(max_depth = max_depth, class_weight = class_weight)
    elif (model_name == 'logistic'):
        model = LogisticRegression()
    elif (model_name == 'adaboost'):
        model = AdaBoostClassifier(n_estimators = n_estimators, learning_rate = learning_rate)
        
    model.fit(X_train, y_train)
    stats, threshes, plot = predict_and_save(X_train, X_tune, y_train, y_tune, model, MODELS_DIR, model_name = model_name)
    
    if (model_name == 'decision_tree'):
        plt.figure(figsize=(50, 40))
        plot_tree(model, feature_names=feature_names, filled=True)
        
    # return stats, threshes, fig, model
    return stats, threshes, plot, model


#---
def fit_other_models(model_name, X_train, X_tune, y_train, y_tune, feature_names, MODELS_DIR,
                     seed_ = 13, verbose = True,
                     listlengthprior=2, max_iter=10000, class1label="IwI",
                     max_rules=4,
                     max_depth=9, class_weight={0: 1, 1: 100}, criterion='neg_corr'):
    
    np.random.seed(seed_)
    
    if model_name == 'bayesian_rule_list':
        model = imodels.BayesianRuleListClassifier(listlengthprior=listlengthprior, max_iter=max_iter, class1label=class1label, verbose=False)
        model.fit(X_train, y_train, feature_names=feature_names)
        if verbose: print(model)
        
    elif model_name == 'rulefit':
        model = imodels.RuleFitRegressor(max_rules=4)
        model.fit(X_train, y_train, feature_names=feature_names)
        if verbose: print(model.visualize())

    elif model_name == 'grl':
        model = imodels.GreedyRuleListClassifier(max_depth=max_depth, class_weight=class_weight, criterion=criterion)
        model.fit(X_train, y_train, feature_names=feature_names) #, verbose=False)
        if verbose: print(model)
        
    stats, threshes, plot = predict_and_save(X_train, X_tune, y_train, y_tune, model, MODELS_DIR, model_name=model_name)

    return stats, threshes, plot, model


#---
def plot_metrics(suffix, MODELS_DIR, title=None, fs=15):
    for fname in sorted(os.listdir(MODELS_DIR)):
        if 'pkl' in fname:
            if not fname[:-4] == 'rf':
                r = pkl.load(open(oj(MODELS_DIR, fname), 'rb'))
                threshes = np.array(r['threshes' + suffix])
                sens = np.array(r['sens' + suffix])
                spec = np.array(r['spec' + suffix])
                
                plt.plot(100 * sens, 100 * spec, 'o-', label=fname[:-4], alpha=0.6, markersize=3)
                plt.xlabel('Sensitivity (%)', fontsize=fs)
                plt.ylabel('Specificity (%)', fontsize=fs)
                
                s = suffix[1:]
                if title is None:
                    plt.title(f'{s}\n{data_sizes[s][0]} IAI-I / {data_sizes[s][1]}')
                else:
                    plt.title(title, fontsize=fs)

                # print best results
                if suffix == '_test2':
                    idxs = (sens > 0.95) & (spec > 0.43)
                    if np.sum(idxs) > 0:
                        idx_max = np.argmax(spec[idxs])
                        print(fname, f'{100 * sens[idxs][idx_max]:0.2f} {100 * spec[idxs][idx_max]:0.2f}')

    if suffix == '_test2':
        plt.plot(96.77, 43.98, 'o', color='black', label='Original CDR', ms=4)
    else:
        plt.plot(97.0, 42.5, 'o', color='black', label='Original CDR', ms=4)
    plt.grid()

    
#---
def print_metrics(suffix, X_train, X_tune, y_train, y_tune, MODELS_DIR):
    vals = {s: [] for s in ['sens', 'spec', 'ppv', 'npv', 'lr+', 'lr-', 'brier_score', 'f1']}
    fnames = []
    for fname in sorted(os.listdir(MODELS_DIR)):
        print(fname)
        if 'pkl' in fname:
            if not fname[:-4] == 'rf':
                r = pkl.load(open(oj(MODELS_DIR, fname), 'rb'))
                threshes = np.array(r['threshes' + suffix])
                m = r['model']

                # add more stats
                for x, y, suff in zip([X_train, X_tune],
                                      [y_train, y_tune],
                                      ['_train', '_tune']):
                        
                    if suff == suffix:
                        if suffix == '_tune':
                            stats, threshes, plot = validation_new.all_stats_curve(y, m.predict_proba(x)[:, 1], plot= True)
                        else:
                            stats, threshes = validation_new.all_stats_curve(y, m.predict_proba(x)[:, 1], plot= False)
                        
                        preds_proba = m.predict_proba(x)[:, 1]
                        brier_score = metrics.brier_score_loss(y, preds_proba)

                # pick best vals
                sens = np.array(r['sens' + suffix])
                spec = np.array(r['spec' + suffix])
                best_idx = np.argmax(5 * sens + spec)
                for k in vals.keys():
                    if not k == 'brier_score':
                        #                         print('k', k)
                        vals[k].append(stats[k][best_idx])
                vals['brier_score'].append(brier_score)
                fnames.append(fname[:-4])
                
    stats = pd.DataFrame.from_dict(vals)
    stats.index = fnames
    return (stats).round(5).transpose()