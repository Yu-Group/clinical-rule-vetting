from os.path import join as oj

import numpy as np
import os
import random
import pandas as pd
import re
from tqdm import tqdm
from typing import Dict
from joblib import Memory

import rulevetting
import rulevetting.api.util
from rulevetting.projects.csi_pecarn import helper
from rulevetting.templates.dataset import DatasetTemplate

def find_best(data, v_list, method = "gini"):
    
    '''
    find the best one to split the data from a variable list
    
    Parameters:
    data: same structure as what we get from Dataset().get_data()[0]
    v_list: names of variable names we are considering
    '''
    
    v = len(v_list)
    n = data.shape[0]
    
    score = [1]*v
    
    for i in range(v):
        
        variable = v_list[i]
        nv = data[data[variable] == 1].shape[0]

        if nv == 0:
            score[i] = 2
        else:
            v1c1 = data[(data[variable] == 1) & (data['csi_injury'] == 1)].shape[0]
            v1c0 = data[(data[variable] == 1) & (data['csi_injury'] == 0)].shape[0]
            if (v1c1+v1c0) == 0:
                p1 = 0
            else:
                p1 = v1c1/(v1c1+v1c0)
            
            v0c1 = data[(data[variable] == 0) & (data['csi_injury'] == 1)].shape[0]
            v0c0 = data[(data[variable] == 0) & (data['csi_injury'] == 0)].shape[0]
            if (v0c1+v0c0) == 0:
                p2 = 0
            else:
                p2 = v0c1/(v0c1+v0c0)
            
            if method == 'gini':
                score[i] = (v1c1+v1c0)/n * p1 * (1-p1) + (v0c1+v0c0)/n * p2 * (1-p2)
                
            elif method == 'semi_gini':
                score[i] = 1-p1
            
        # print(variable, p1, score[i])
    
    ind = score.index(min(score))
    variable_best = v_list[ind]
    v_list.remove(variable_best)
    data_update = data[data[variable_best] == 0]
    
    return [variable_best, v_list, data_update]

def find_best_two(data, v_list, method = "gini"):

    '''
    find the best one to split the data from a variable list
    
    Parameters:
    data: same structure as what we get from Dataset().get_data()[0]
    v_list: names of variable names we are considering
    '''
    
    v = len(v_list)
    n = data.shape[0]
    
    score = [1]*v
    for i in range(v):
        variable = v_list[i]
        v1c1 = data[(data[variable] == 1) & (data['csi_injury'] == 1)].shape[0]
        v1c0 = data[(data[variable] == 1) & (data['csi_injury'] == 0)].shape[0]
        if (v1c1+v1c0) == 0:
            p1 = 1/2
        else:
            p1 = v1c1/(v1c1+v1c0)
        v0c1 = data[(data[variable] == 0) & (data['csi_injury'] == 1)].shape[0]
        v0c0 = data[(data[variable] == 0) & (data['csi_injury'] == 0)].shape[0]
        if (v0c1+v0c0) == 0:
            p2 = 1/2
        else:
            p2 = v0c1/(v0c1+v0c0)
        if method == 'gini':
            score[i] = (v1c1+v1c0)/n * p1 * (1-p1) + (v0c1+v0c0)/n * p2 * (1-p2)  
        elif method == 'semi_gini':
            score[i] = 1-p1   
        # print(variable, p1, score[i])
    ind = score.index(min(score))
    variable_best = v_list[ind]
    v_list.remove(variable_best)
    data_update = data[data[variable_best] == 0]
    
    # find the second rule
    
    data_selected = data[data[variable_best] == 1]
    n0 = data_selected.shape[0]
    
    if n0 == 0:
        variable_best_two = [variable_best, "no observations"]
        return [variable_best_two, v_list, data_update]
    
    p0 = data_selected[data_selected['csi_injury'] == 1].shape[0]/n0
    score0 = p0*(1-p0)
    
    score = [1]*(v-1)
    for i in range(v-1):
        variable = v_list[i]
        
        v1c1 = data_selected[(data_selected[variable] == 1) & (data_selected['csi_injury'] == 1)].shape[0]
        v1c0 = data_selected[(data_selected[variable] == 1) & (data_selected['csi_injury'] == 0)].shape[0]
        if (v1c1+v1c0) == 0:
            p1 = 1/2
        else:
            p1 = v1c1/(v1c1+v1c0)
        v0c1 = data_selected[(data_selected[variable] == 0) & (data_selected['csi_injury'] == 1)].shape[0]
        v0c0 = data_selected[(data_selected[variable] == 0) & (data_selected['csi_injury'] == 0)].shape[0]
        if (v0c1+v0c0) == 0:
            p2 = 1/2
        else:
            p2 = v0c1/(v0c1+v0c0)
    
        # use gini index -- will improve specificity but great hurt sensitivity
        # score[i] = (v1c1+v1c0)/n0 * p1 * (1-p1) + (v0c1+v0c0)/n0 * p2 * (1-p2)  
        score[i] = p2
    
    if v > 1:

        min_score = min(score)
        if min_score > 0.1:
            variable_best_two = [variable_best, "no need"]
        else:
            ind = score.index(min_score)
            variable_best_two = [variable_best, v_list[ind]]
    else:

        variable_best_two = [variable_best, "no variable"]
        
    return [variable_best_two, v_list, data_update]


def make_decision_ob(observation, v_list):
    
    '''
    make decision by v_list for one single observation
    '''

    n = len(v_list)
    for i in range(n):
        
        v0 = v_list[i][0]
        v1 = v_list[i][1]
        
        if (observation[v0].item() == 1):
            if (v1 in ["no need", 'no observations']):
                return 1
            elif (observation[v1].item() == 1):
                return 1
            
    return 0
        

def make_decision_data(data, v_list):

    '''
    make decision by v_list for one dataframe
    '''
    
    n = data.shape[0]
    decision = [0]*n
    for i in range(n):
        observation = data.iloc[[i]]
        decision[i] = make_decision_ob(observation, v_list)
    return decision

def evaluate_vlist(data, v_list, method = 'one'):

    '''
    generate the sensitity and sepecifity for v_list on data
    '''
    
    data0 = pd.DataFrame({'csi_injury': data['csi_injury']})
    
    if method == "one":
        indicator = data[v_list].sum(axis = 1)
        data0['pred'] = [1 if (i > 0) else 0 for i in indicator]
    if method == "two":
        data0['pred'] = make_decision_data(data, v_list)

    TN = data0[ (data0['pred'] == 0) & (data0['csi_injury'] == 0)].shape[0]
    TP = data0[ (data0['pred'] == 1) & (data0['csi_injury'] == 1)].shape[0]
    FN = data0[ (data0['pred'] == 0) & (data0['csi_injury'] == 1)].shape[0]
    FP = data0[ (data0['pred'] == 1) & (data0['csi_injury'] == 0)].shape[0]
    
    sensitivity = TP/(FN+TP)
    specificity = TN/(FP+TN)
    
    return [sensitivity, specificity]

def simple_tree(data_list, tree_method, select_method):

    '''
    1. train model on data_list training set by tree_method and select_method
    2. evaluate model on data_list tuning set
    
    return 
    1. the variablelist for the fitted model
    2. evaluation list of TPR and FPR on training set
    3. evaluation list of TPR and FPR on tuning set

    '''
    
    data = data_list[0].copy()
    v_list = list(data.columns)
    v_list.remove('csi_injury')
    variable_rank = []
    if tree_method == 'one':
        while len(v_list) > 0:
            # result = find_best_two(data,v_list,method = "semi_gini")
            result = find_best(data,v_list,select_method)
            variable_rank.append(result[0])
            v_list = result[1]
            data = result[2]
    elif tree_method == 'two':
        while len(v_list) > 0:
            result = find_best_two(data,v_list,select_method)
            # result = find_best(data,v_list,method = select_method)
            variable_rank.append(result[0])
            v_list = result[1]
            data = result[2]

    l = len(variable_rank)
    ind = range(l)
    TPR = [0]*l
    FPR = [0]*l
    for i in ind:
        r = evaluate_vlist(data_list[0],variable_rank[0:(i+1)], tree_method)
        TPR[i] = r[0]
        FPR[i] = 1- r[1]
    d = {'num': ind, 'TPR': TPR, 'FPR': FPR}
    evaluation_training = pd.DataFrame(data = d)

    l = len(variable_rank)
    ind = range(l)
    TPR = [0]*l
    FPR = [0]*l
    for i in ind:
        r = evaluate_vlist(data_list[1],variable_rank[0:(i+1)], tree_method)
        TPR[i] = r[0]
        FPR[i] = 1- r[1]
    d = {'num': ind, 'TPR': TPR, 'FPR': FPR}
    evaluation_tuning = pd.DataFrame(data = d)

    return [variable_rank, evaluation_training, evaluation_tuning]












