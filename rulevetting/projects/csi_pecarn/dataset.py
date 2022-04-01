import glob
from os.path import join as oj
from unittest import result

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict

import rulevetting
import rulevetting.api.util
from rulevetting.projects import one_hot_encode_df
from rulevetting.projects.iai_pecarn import helper
from rulevetting.templates.dataset import DatasetTemplate


MERGE_KEYS = ['SITE', 'CaseID', 'StudySubjectID']


class Dataset(DatasetTemplate):
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        fnames = sorted(glob.glob(f'{raw_data_path}/*'))
        dfs = [pd.read_csv(fname, encoding="ISO-8859-1") for fname in fnames]
        #print(fnames)

        clean_key_col_names = lambda df: df.rename(columns={'site': 'SITE',
                                                            'caseid': 'CaseID',
                                                            'studysubjectid': 'StudySubjectID'})
        # Build on AnalysisVariable
        result_df = dfs[0].copy()

        key_df = dfs[0][MERGE_KEYS]
        # Read clinical presentation
        field_df = key_df.merge(clean_key_col_names(dfs[1]), how='left',
                                on=MERGE_KEYS)
        outside_df = key_df.merge(clean_key_col_names(dfs[2]), how='left',
                                on=MERGE_KEYS)
        site_df = key_df.merge(clean_key_col_names(dfs[3]), how='left',
                                on=MERGE_KEYS)
        
        # + Is patient sent by EMS?
        result_df['is_ems'] = 0
        result_df.loc[(field_df['FieldDocumentation'].isin(['EMS', 'NR'])), 'is_ems'] = 1
        
        # + Patient position
        position_df = pd.get_dummies(field_df['PatientsPosition'], prefix='Position')
        result_df = result_df.merge(
            pd.concat([field_df[MERGE_KEYS], position_df], axis=1),
            how='left', on=MERGE_KEYS)

        if kwargs['fillna']:
            # + Compaint of pain in other region
            result_df['PtCompPainHead'] = site_df['PtCompPainHead'].fillna(0)
            result_df['PtCompPainFace'] = site_df['PtCompPainFace'].fillna(0)
            result_df['PtCompPainExt'] = site_df['PtCompPainExt'].fillna(0)
            result_df['PtCompPainTorsoTrunk'] = site_df[['PtCompPainChest', 'PtCompPainBack', 'PtCompPainFlank', 'PtCompPainAbd', 'PtCompPainPelvis']].fillna(0).sum(axis = 1)
            result_df.loc[result_df['PtCompPainTorsoTrunk'] > 0, 'PtCompPainTorsoTrunk'] = 1

            result_df['PtCompPainHead2'] = result_df['PtCompPainHead'] + field_df['PtCompPainHead'].fillna(0) + outside_df['PtCompPainHead'].fillna(0)
            result_df.loc[result_df['PtCompPainHead2'] > 0, 'PtCompPainHead2'] = 1
            result_df['PtCompPainFace2'] = result_df['PtCompPainFace'] + field_df['PtCompPainFace'].fillna(0) + outside_df['PtCompPainFace'].fillna(0)
            result_df.loc[result_df['PtCompPainFace2'] > 0, 'PtCompPainFace2'] = 1
            result_df['PtCompPainExt2'] = result_df['PtCompPainExt'] + field_df['PtCompPainExt'].fillna(0) + outside_df['PtCompPainExt'].fillna(0)
            result_df.loc[result_df['PtCompPainExt2'] > 0, 'PtCompPainExt2'] = 1
            result_df['PtCompPainTorsoTrunk2'] = result_df['PtCompPainTorsoTrunk'] + field_df[['PtCompPainChest', 'PtCompPainBack', 'PtCompPainFlank', 'PtCompPainAbd', 'PtCompPainPelvis']].fillna(0).sum(axis = 1) + outside_df[['PtCompPainChest', 'PtCompPainBack', 'PtCompPainFlank', 'PtCompPainAbd', 'PtCompPainPelvis']].fillna(0).sum(axis = 1)
            result_df.loc[result_df['PtCompPainTorsoTrunk2'] > 0, 'PtCompPainTorsoTrunk2'] = 1        
            
            # + Tenderness in other region
            result_df['PtTenderHead'] = site_df['PtTenderHead'].fillna(0)
            result_df['PtTenderFace'] = site_df['PtTenderFace'].fillna(0)
            result_df['PtTenderExt'] = site_df['PtTenderExt'].fillna(0)
            result_df['PtTenderTorsoTrunk'] = site_df[['PtTenderChest', 'PtTenderBack', 'PtTenderFlank', 'PtTenderAbd', 'PtTenderPelvis']].fillna(0).sum(axis = 1)
            result_df.loc[result_df['PtTenderTorsoTrunk'] > 0, 'PtTenderTorsoTrunk'] = 1

            result_df['PtTenderHead2'] = result_df['PtTenderHead'] + field_df['PtTenderHead'].fillna(0) + outside_df['PtTenderHead'].fillna(0)
            result_df.loc[result_df['PtTenderHead2'] > 0, 'PtTenderHead2'] = 1
            result_df['PtTenderFace2'] = result_df['PtTenderFace'] + field_df['PtTenderFace'].fillna(0) + outside_df['PtTenderFace'].fillna(0)
            result_df.loc[result_df['PtTenderFace2'] > 0, 'PtTenderFace2'] = 1
            result_df['PtTenderExt2'] = result_df['PtTenderExt'] + field_df['PtTenderExt'].fillna(0) + outside_df['PtTenderExt'].fillna(0)
            result_df.loc[result_df['PtTenderExt2'] > 0, 'PtTenderExt2'] = 1
            result_df['PtTenderTorsoTrunk2'] = result_df['PtTenderTorsoTrunk'] + field_df[['PtTenderChest', 'PtTenderBack', 'PtTenderFlank', 'PtTenderAbd', 'PtTenderPelvis']].fillna(0).sum(axis = 1) + outside_df[['PtTenderChest', 'PtTenderBack', 'PtTenderFlank', 'PtTenderAbd', 'PtCompPainPelvis']].fillna(0).sum(axis = 1)
            result_df.loc[result_df['PtTenderTorsoTrunk2'] > 0, 'PtTenderTorsoTrunk2'] = 1   
        

        # + Intervention after inital evaluation?
        if kwargs['include_intervention']:
            result_df['Immobilization'] = 0
            result_df.loc[(site_df['CervicalSpineImmobilization'].isin([1, 2])), 'Immobilization'] = 1
            result_df['Immobilization2'] = result_df['Immobilization'].copy()
            result_df.loc[(outside_df['CervicalSpineImmobilization'].isin(['YD', 'YND'])), 'Immobilization2'] = 1

            result_df['MedsRecd'] = 0
            result_df.loc[(site_df['MedsRecdPriorArrival'] == 'Y'), 'MedsRecd'] = 1
            result_df['MedsRecd2'] = result_df['MedsRecd'].copy()
            result_df.loc[(outside_df['MedsRecdPriorArrival'] == 'Y'), 'MedsRecd2'] = 1
            
            result_df['ArrPtIntub'] = 0
            result_df.loc[(site_df['ArrPtIntub'] == 'Y'), 'ArrPtIntub'] = 1
            result_df['ArrPtIntub2'] = result_df['ArrPtIntub'].copy()
            result_df.loc[(outside_df['ArrPtIntub'] == 'Y'), 'ArrPtIntub2'] = 1
            
        
        # result_df['Precaution'] = 0
        # result_df.loc[(site_df['CSpinePrecautions'].isin(['YD', 'YND'])), 'Precaution'] = 1
        # result_df['Precaution2'] = result_df['Precaution']
        # result_df.loc[(outside_df['CervicalSpinePrecautions'].isin(['YD', 'YND'])), 'Precaution2'] = 1
        
        # + Gender and age
        demog_df = clean_key_col_names(dfs[4])
        gender_df = pd.get_dummies(demog_df['Gender'], prefix='gender').drop(columns='gender_ND')
        # agegroup_df = pd.get_dummies(pd.cut(demog_df['AgeInYears'], bins=[0, 2, 6, 12, 16],
        #                                     labels=['infant', 'preschool', 'school_age', 'adolescents'],
        #                                     include_lowest=True), prefix='age')
        # agegroup_df['AgeInYears'] = demog_df['AgeInYears']

        result_df = result_df.merge(
            pd.concat([demog_df[MERGE_KEYS + ['AgeInYears']], gender_df], axis=1),
            how='left', on=MERGE_KEYS)
        result_df = result_df.loc[(result_df['gender_F'] == 1) | (result_df['gender_M'] == 1)].drop(columns='gender_M').reset_index(drop=True)

        site_meta_keys = ['EDDisposition', 'IntervForCervicalStab', 'IntervForCervicalStabSCollar', 'IntervForCervicalStabRCollar', 'IntervForCervicalStabBrace', 'IntervForCervicalStabTraction', 'IntervForCervicalStabSurgical', 'IntervForCervicalStabHalo', 'IntervForCervicalStabIntFix', 'IntervForCervicalStabIntFixtxt', 'IntervForCervicalStabOther', 'IntervForCervicalStabOthertxt', 'LongTermRehab', 'OutcomeStudySiteNeuro', 'OutcomeStudySiteMobility', 'OutcomeStudySiteMobility1', 'OutcomeStudySiteMobility2', 'OutcomeStudySiteBowel', 'OutcomeStudySiteUrine']
        result_df = result_df.merge(site_df[MERGE_KEYS + site_meta_keys], how='left', on=MERGE_KEYS)
        return result_df

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # drop cols with vals missing this percent of the time
        # df = cleaned_data.dropna(axis=1, thresh=(1 - kwargs['frac_missing_allowed']) * cleaned_data.shape[0])

        # drop rows 
        # df = cleaned_data.dropna(axis=0, thresh=(1 - kwargs['frac_missing_allowed']) * cleaned_data.shape[1])
        df = cleaned_data
        
        # impute missing values
        # fill in values for some vars from unknown -> None
        # df = cleaned_data.dropna(axis=0)

        # Impute: Baseline fill NA with median
        # df = df.fillna(df.median()) 

        if kwargs['fillna']:
            # Impute: Use domain knowledge
            liberal_feats = ['FocalNeuroFindings', 'FocalNeuroFindings2', 'Torticollis', 'Torticollis2', 
                            'SubInj_Head', 'SubInj_Face', 'SubInj_Ext', 'SubInj_TorsoTrunk', 'subinj_Head2', 'subinj_Face2', 'subinj_Ext2', 'subinj_TorsoTrunk2',
                            'Predisposed', 
                            'HighriskMVC', 'HighriskDiving', 'HighriskFall', 'HighriskHanging', 'HighriskHitByCar', 'HighriskOtherMV', 'AxialLoadAnyDoc', 'axialloadtop', 'Clotheslining']
            conserv_feats = ['LOC']
            unclear_feats = ['AlteredMentalStatus', 'AlteredMentalStatus2', 'ambulatory', 'PainNeck', 'PainNeck2', 'PosMidNeckTenderness', 'PosMidNeckTenderness2', 'TenderNeck', 'TenderNeck2']
            df[liberal_feats] = df[liberal_feats].fillna(0)
            df[conserv_feats] = df[conserv_feats].fillna(1)
            unclear_feat_default = kwargs['unclear_feat_default']
            df[unclear_feats] = df[unclear_feats].fillna(unclear_feat_default)
            
            # Impute others to be 0
            df = df.fillna(0)
        
        # drop missing values
        #df = df.dropna(axis=0)
        

       #  # don't use features end with 2
       #  df <- df.filter(regex = '[^2]$', axis = 1)

        # Use only on-site data or also outside + field
        feats1 = ['AlteredMentalStatus', 'FocalNeuroFindings', 'Torticollis', 'PainNeck', 'TenderNeck', 'PosMidNeckTenderness', 'PtCompPainHead', 'PtCompPainFace', 'PtCompPainExt', 'PtCompPainTorsoTrunk', 'PtTenderHead', 'PtTenderFace', 'PtTenderExt', 'PtTenderTorsoTrunk', 'SubInj_Head', 'SubInj_Face', 'SubInj_Ext', 'SubInj_TorsoTrunk', 'Immobilization', 'MedsRecd', 'ArrPtIntub']
        feats2 = ['AlteredMentalStatus2', 'FocalNeuroFindings2', 'Torticollis2', 'PainNeck2', 'TenderNeck2', 'PosMidNeckTenderness2', 'PtCompPainHead2', 'PtCompPainFace2', 'PtCompPainExt2', 'PtCompPainTorsoTrunk2', 'PtTenderHead2', 'PtTenderFace2', 'PtTenderExt2', 'PtTenderTorsoTrunk2', 'subinj_Head2', 'subinj_Face2', 'subinj_Ext2', 'subinj_TorsoTrunk2', 'Immobilization2', 'MedsRecd2', 'ArrPtIntub2']
        feats1 = list(set(feats1) & set(df.columns))
        feats2 = list(set(feats2) & set(df.columns))
#         if kwargs['only_site_data'] == True:
#             df = df.drop(columns = feats2)
#         elif kwargs['only_site_data'] == False:
#             df = df.drop(columns = feats1)
        if kwargs['only_site_data'] == 1:
            df = df.drop(columns = feats2)
        elif kwargs['only_site_data'] == 2:
            df = df.drop(columns = feats1)

        # Only analysisVariable
        if not kwargs['augmented_features']:
            feat_augmented = list(set(['PtCompPainHead', 'PtCompPainFace',
           'PtCompPainExt', 'PtCompPainTorsoTrunk', 'PtTenderHead',
           'PtTenderFace', 'PtTenderExt2', 'PtTenderTorsoTrunk',
           'Immobilization', 'MedsRecd', 'ArrPtIntub',
           'Position_IDEMS', 'Position_L', 'Position_ND', 'Position_PA',
           'Position_S', 'Position_W', 'PtCompPainHead2', 'PtCompPainFace2',
           'PtCompPainExt2', 'PtCompPainTorsoTrunk2', 'PtTenderHead2',
           'PtTenderFace2', 'PtTenderExt2', 'PtTenderTorsoTrunk2',
           'Immobilization2', 'MedsRecd2', 'ArrPtIntub2', 'age_infant',
           'age_preschool', 'age_school_age', 'age_adolescents']) & set(df.columns))
            df = df.drop(columns = feat_augmented)
            
        # only one type of control
        if kwargs['use_control_type'] == 'ran':
            df = df[df['ControlType'].isin(['case', 'ran'])]
        elif kwargs['use_control_type'] == 'moi':
            df = df[df['ControlType'].isin(['case', 'moi'])]
        elif kwargs['use_control_type'] == 'ems':         
            df = df[df['ControlType'].isin(['case', 'ems'])]

        df.loc[:, 'outcome'] = (df['ControlType'] == 'case').astype(int)
        
        return df

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # add engineered featuures
        # df = helper.derived_feats(preprocessed_data)

        # convert feats to dummy
        # df = pd.get_dummies(preprocessed_data, dummy_na=True)  # treat na as a separate category

        # remove any col that is all 0s
        df = preprocessed_data.loc[:, (preprocessed_data != 0).any(axis=0)]

        # remove the _no columns
        # if kwargs['drop_negative_columns']:
         #    df.drop([k for k in df.keys() if k.endswith('_no')], inplace=True)

        # remove (site), case ID, subject ID, control type
        df_encoded = one_hot_encode_df(df, numeric_cols=self.get_meta_keys())

        df_encoded.insert(
            len(df_encoded.columns) - 1, 'outcome', df_encoded.pop('outcome'))

        return df_encoded

    def split_data(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """Split into 3 sets: training, tuning, testing.
        Do not modify (to ensure consistent test set).
        Keep in mind any natural splits (e.g. hospitals).
        Ensure that there are positive points in all splits.
        Parameters
        ----------
        preprocessed_data
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls
        Returns
        -------
        df_train
        df_tune
        df_test
        """
        # if kwargs['data_split'] == 'split_by_site':
        #     sites = np.arange(1, 18)
        #     np.random.seed(42)
        #     np.random.shuffle(sites)
        #     site_split = np.split(sites, [9, 13])
        #     split = tuple([preprocessed_data[preprocessed_data['SITE'].isin(site_split[0])],
        #                  preprocessed_data[preprocessed_data['SITE'].isin(site_split[1])],
        #                  preprocessed_data[preprocessed_data['SITE'].isin(site_split[2])]])
        # elif kwargs['data_split'] == 'random_split':
        #     split = tuple(np.split(
        #     preprocessed_data.sample(frac=1, random_state=42),
        #     [int(.6 * len(preprocessed_data)),  # 60% train
        #      int(.8 * len(preprocessed_data))]  # 20% tune, 20% test
        # ))

        site_as_int = preprocessed_data['SITE'].astype(int)
        
        sites = np.arange(1, 18)
        np.random.seed(42)
        np.random.shuffle(sites)
        site_split = np.split(sites, [4])
        split = tuple(np.split(
            preprocessed_data[site_as_int.isin(site_split[1])].sample(frac=1, random_state=42), 
            [int(.75 * sum(site_as_int.isin(site_split[1])))]) +  # 60% train 20% tune
                      [preprocessed_data[site_as_int.isin(site_split[0])]]) # 20% test
        
        # sites = np.arange(1, 18)
        # np.random.seed(42)
        # np.random.shuffle(sites)
        # site_split = np.split(sites, [9, 13])
        # split = tuple([preprocessed_data[preprocessed_data['SITE'].isin(site_split[0])],
        #               preprocessed_data[preprocessed_data['SITE'].isin(site_split[1])],
        #               preprocessed_data[preprocessed_data['SITE'].isin(site_split[2])]])
        
        return split
    
    def get_outcome_name(self) -> str:
        return 'csi'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'csi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        site_keys = ['EDDisposition', 'IntervForCervicalStab', 'IntervForCervicalStabSCollar', 'IntervForCervicalStabRCollar', 'IntervForCervicalStabBrace', 'IntervForCervicalStabTraction', 'IntervForCervicalStabSurgical', 'IntervForCervicalStabHalo', 'IntervForCervicalStabIntFix', 'IntervForCervicalStabIntFixtxt', 'IntervForCervicalStabOther', 'IntervForCervicalStabOthertxt', 'LongTermRehab', 'OutcomeStudySiteNeuro', 'OutcomeStudySiteMobility', 'OutcomeStudySiteMobility1', 'OutcomeStudySiteMobility2', 'OutcomeStudySiteBowel', 'OutcomeStudySiteUrine']
        return site_keys + ['SITE', 'StudySubjectID', 'ControlType', 'CaseID', 'AgeInYears']  # keys which are useful but not used for prediction

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        return {
            'clean_data': {
                # Include features about clinical intervention received before arrival
                #'include_intervention': [True, False],
                'include_intervention': [False, True], # after stability analysis new jcall
                'fillna': [True, False]
            },
            'preprocess_data': {
                # for unclear features whether to impute conservatively or liberally
                'unclear_feat_default': [0, 1], 
                # Whether to use only data from the study site or also include field and outside hospital data
#                 'only_site_data': [0, 1, 2],
                'only_site_data': [2, 1],
                # Whether to use augmented features or original AnalysisVariables
                #'augmented_features': [True, False],
                'augmented_features': [True, False, True], # after stability analysis new jcall
                # Use with control group
#                 'use_control_type': ['all', 'ran', 'moi', 'ems']
                'use_control_type': ['all'],
                'fillna': [True, False]
            },
            'extract_features': {
                # whether to drop columns with suffix _no
                'drop_negative_columns': [False],  # default value comes first
            },
            # 'split_data': {
            #     # how do we split data
            #     'data_split': ['split_by_site', 'random_split'],
            # },
        }


if __name__ == '__main__':
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=True)
    print('successfuly processed data\nshapes:',
          df_train.shape, df_tune.shape, df_test.shape,
          '\nfeatures:', list(df_train.columns))
