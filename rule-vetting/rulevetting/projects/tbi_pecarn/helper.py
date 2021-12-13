import os
import numpy as np
import pandas as pd
from rulevetting.projects.tbi_pecarn import dataset
# from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)


def recalculate_gcs(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Many GCSTotal scores do no match their associated metrics; this function fixes these errors.
    '''
    gcs_scores = sum([data[x] for x in data.columns if 'GCS' in x and 'GCSGroup' not in x and x != 'GCSTotal'])
    pd.options.mode.chained_assignment = None
    data['GCSTotal'] = gcs_scores
    return data

def recalc_outcome(data):
    col = data.DeathTBI.copy()
    col[data.HospHead==1] = 1
    col[data.Intub24Head==1] = 1
    col[data.Neurosurgery==1] = 1
    data.PosIntFinal = col
    return data

def remove_patients_needing_CT(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove patients who clearly need a CT scan.
    '''

    data = data[data.FontBulg != 1]
    data = data[data.SFxPalp != 1]
    data = data[data.SFxBas != 1]
    return data
    
def subset_rows(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    '''
    Remove patients for whom:
    • the outcome variable is NA
    • GCS is below 14
    • there are signs of a skull fracture (if specified in kwargs)
    • the anterior fontanelle is bulging
    '''
    # Remove NA outcome
    data = data.dropna(subset=["PosIntFinal"])
    # Recalculate GCS and remove rows with scores below 14
    data = recalculate_gcs(data)
    data = data[data.GCSTotal >= 14]
    if kwargs["only_mildest_trauma"]:
        # Remove patients with signs of a skull fracture or bulging anterior fontanelle
        data = remove_patients_needing_CT(data)
        
    return data

def subset_cols(data: pd.DataFrame, outcome_name, **kwargs) -> pd.DataFrame:
    '''
    Restricts the features of the data frame to only symptoms, demographics, and the outcome variable.
    Subsequently removes rows whose values are missing a certain percentage of the time.
    '''
    # Got rid of SfX, OSI, AgeInMonth, etc
    demographics = ['AgeTwoPlus']
    symptoms = ['High_impact_InjSev', 'Amnesia_verb', 'LocLen', 'Seiz', 'SeizOccur', 'SeizLen', 'ActNorm',
            'HA_verb', 'HASeverity', 'HAStart', 'Vomit', 'VomitNbr', 'VomitStart', 'VomitLast', 'Dizzy', 'GCSEye',
            'GCSVerbal', 'GCSMotor', 'GCSTotal', 'AMS', #'AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth', 
            'Hema', 'HemaLoc', 'HemaSize', 'Clav', 'ClavFace', 'ClavNeck',
            'ClavFro', 'ClavOcc', 'ClavTem', 'Drugs', 'LOCSeparate', 'NeuroD']#, 
#            'NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth']
    if kwargs["only_mildest_trauma"] == False:
        symptoms.extend(['SFxPalp', 'FontBulg', 'SFxBas'])
    
    subset = []
    subset.extend(demographics)
    subset.extend(symptoms)
    subset.extend([outcome_name])
    data = data[subset]
    # Remove features with over frac percent of vals missing
    frac = kwargs["frac_missing_allowed"]
    data = data.dropna(axis=1, thresh=(1 - frac) * data.shape[0])

    return data

def impute_data(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    '''
    Imputes missing data with K-Nearest Neighbors (slow) or median.
    '''
    key = "imputation"
    if kwargs[key] == "KNN":
        return impute_knn(data)
    elif kwargs[key] == "median":
        data_imputed = data.fillna(data.median())
        return data_imputed
    elif kwargs[key] == "none":
        return data
    else:
        print("\"imputation\" must be KNN, median, or none.")

def binarize_data(data: pd.DataFrame, outcome_name, **kwargs) -> pd.DataFrame:
    '''
    Encodes each categorical variable as multiple binary columns.
    Also changes column name of outcome variable from "PosIntFinal" to "outcome."
    '''
    # Transform categorical variables to binary
    column_names = set(data.columns) - {outcome_name}
    numeric_cols = {"GCSTotal"}
    categorical_cols = column_names - numeric_cols

    # Convert each categorical variable to a set of binary columns
    # One for each level, including na (different from nan)
    data_binarized = data[numeric_cols].copy()
    levels_df = {col: data[col].unique() for col in categorical_cols}
    for col in categorical_cols:
        levels = sorted(levels_df[col])# e.g. 0, 1, 92
        for level in levels:
            try:
                if level < 90:
                    name = str(round(level))
                    newcol = col + "_" + name
                    data_binarized[newcol] = (data[col]==level).astype(int)
            except:# missing value
                pass
    # Add binary outcome variable
    data_binarized[outcome_name] = data[outcome_name]
    # Transform GCSTotal to binary
    data_binarized["GCS_14"] = (data.GCSTotal==14).astype(int)
    return data_binarized

def change_outcome_name(data: pd.DataFrame, outcome_name) -> pd.DataFrame:
    data["outcome"] = data[outcome_name]
    data = data.drop(columns = [outcome_name])
    return data

def process_and_save(data_path, filename, **kwargs):
    # Loads dataset
    dset = dataset.Dataset()
    df_pub = dset.clean_data(data_path)
    preprocessed_data = dset.preprocess_data(df_pub, **kwargs["preprocess_data"])
    # Saves data
    fp = os.path.join(data_path, filename)
    preprocessed_data.to_csv(fp, index=False)
    return preprocessed_data

def get_features(data_path, **kwargs):
    # Preprocesses data except binarization
    dset = dataset.Dataset()
    data = dset.clean_data(data_path)
    outcome_name = dset.get_outcome_name()
    data_subset = subset_data(data, outcome_name)
    data_subset2 = remove_data(data_subset, **kwargs)
    data_imputed = impute_data(data_subset2, **kwargs)
    features = data_imputed.columns
    return features
    
def default_judgement_calls_preprocessing():
    default_kwargs = default_judgement_calls()
    default_kwargs_preprocessing = default_kwargs["preprocess_data"]
    return default_kwargs_preprocessing

def default_judgement_calls():
    dset = dataset.Dataset()
    kwargs = dset.get_judgement_calls_dictionary()
    default_kwargs = {}
    for key in kwargs.keys():
        func_kwargs = kwargs[key]
        default_kwargs[key] = {k: func_kwargs[k][0]  # first arg in each list is default
                               for k in func_kwargs.keys()}
    return default_kwargs

def derived_feats(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    # Final feature selection: high weight, low iteration, iterative, only positive correlation
    # Probably the best feature selection metric we have currently
    important_variables_iter = list(preprocessed_data.columns)
    important_variables_iter.remove("outcome")
    outcome = preprocessed_data['outcome']
    while True:
        selector = SelectFromModel(estimator= LogisticRegression(random_state=0, class_weight= {0:1, 1:500}, max_iter = 100),
                                       threshold = f'0.75*mean').fit(preprocessed_data[important_variables_iter], outcome)

        important_variables_iter = [list(preprocessed_data[important_variables_iter].columns)[i] 
                               for i in range(len(preprocessed_data[important_variables_iter].columns)) 
                               if selector.get_support()[i] == True and selector.estimator_.coef_[0][i] > 0]

        if len(important_variables_iter) <= 30:
            break
    important_variables_iter.append('outcome')
    return preprocessed_data[important_variables_iter]
    
def impute_knn(data):
    '''
    Imputes missing values by finding similarity of each row with 10 randomly sampled
    others, and choosing the values of the closest one.
    '''
    n = data.shape[0]
    for i in range(n):
        row_i = data.iloc[i]
        row_i_na = row_i.isna()
        best_j = -1
        most_same = -1
        if sum(row_i_na) > 0:
            j_vals = np.random.choice(n, 10, replace=False)
            for j in j_vals:
                if j != i:
                    row_j = data.iloc[j]
                    row_j_na = row_j.isna()
                    neither_na = [not a and not b for a, b in zip(row_i_na, row_j_na)]
                    frac_same = sum(row_i[neither_na] == row_j[neither_na])/len(neither_na)
                    if frac_same > most_same:
                        best_j = j
                        most_same = frac_same
            # Row i missing, row j full --> fill with row j
            i_na_j_full = [a and not b for a, b in zip(row_i_na, row_j_na)]
            for idx in np.where(i_na_j_full)[0]:
                data.iloc[i, idx] = row_j[idx]

    # Fill values that are still NA with median
    data_imputed = data.fillna(data.median())
    return data_imputed

def meta_key_colnames(data):# Unused
    dset = dataset.Dataset()
    meta_keys = dset.get_meta_keys()
    meta_key_cols = []
    for col in list(data.columns):
        for key in meta_keys:
            if key in col:
                meta_key_cols.append(col)
    return meta_key_cols

def get_modeling_features(data):# Unused
    dset = dataset.Dataset()
    modeling_features = set(data.columns) - set(meta_key_colnames(data)) - {"outcome"}
    return modeling_features

def split_by_age(data, **kwargs):
    if kwargs['split_by_age'] != "no":
        if kwargs['split_by_age'] == "older":
            data = data[data.AgeTwoPlus_2 == 1]
        elif kwargs['split_by_age'] == "younger":
            data = data[data.AgeTwoPlus_2 == 0]
        else:
            print("Error: split_by_age must be no, older, or younger.")
    return data

def combine_features(data):
    cols_to_remove = ['LocLen_2', 'LocLen_3', 'LocLen_4',
                     'SFxPalp_1', 'SFxPalp_2',
                     'LOCSeparate_1', 'LOCSeparate_2',
                     'SeizOccur_2', 'SeizOccur_3',
                     'SeizLen_3', 'SeizLen_4',
                     'VomitStart_3', 'VomitStart_4',
                     'VomitNbr_2', 'VomitNbr_3',
                     'HemaLoc_2', 'HemaLoc_3',
                     'GCSTotal']
    if "LocLen_2" in data.columns:
        # Loss of consciousness greater than 5 seconds
        data['LocLen_2_3_4'] = data['LocLen_2'] | data['LocLen_3'] | data['LocLen_4']
    if "SFxPalp_1" in data.columns:
        # Palpable skull fracture (1) or unclear exam (2)
        data['SFxPalp_1_or_2'] = data['SFxPalp_1'] |  data['SFxPalp_2']
    if "LOCSeparate_1" in data.columns:
        # Yes or suspected
        data['LOCSeparate_1_or_2'] = data['LOCSeparate_1'] |  data['LOCSeparate_2']
    if "SeizOccur_2" in data.columns:
        # Delayed seizure
        data['SeizOccur_2_or_3'] = data['SeizOccur_2'] |  data['SeizOccur_3']
    if "SeizLen_3" in data.columns:
        # Long seizure
        data['SeizLen_3_or_4'] = data['SeizLen_3'] |  data['SeizLen_4']
    if "VomitStart_3" in data.columns:
        # Delayed vomiting
        data['VomitStart_3_or_4'] = data['VomitStart_3'] |  data['VomitStart_4']
    if "VomitNbr_2" in data.columns:
        # Multiple vomiting episodes
        data['VomitNbr_2_or_3'] = data['VomitNbr_2'] |  data['VomitNbr_3']
    if "HemaLoc_2" in data.columns:
        # Multiple vomiting episodes
        data['HemaLoc_2_or_3'] = data['HemaLoc_2'] |  data['HemaLoc_3']
    data = data.drop(columns = cols_to_remove)
    return data

def get_baseline_data(data1, data2, data3):
    # Adds baseline columns
    dset = dataset.Dataset()
    baseline = dset.get_baseline_keys()
    features = []
    for feature in baseline:
        if feature in data1.columns:
            features.append(feature)
    features.append("outcome")
    return data1[features], data2[features], data3[features]
