import glob
from os.path import join as oj

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict
import functools

import rulevetting
import rulevetting.api.util
from rulevetting.projects.iai_pecarn import helper
from rulevetting.templates.dataset import DatasetTemplate


class Dataset(DatasetTemplate):
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        fnames = sorted(glob.glob(f'{raw_data_path}/*'))
        dfs = [pd.read_csv(fname) for fname in fnames]
        
        # Unify the column names
        dfs[1] = dfs[1].rename(columns={'site': 'SITE', 'caseid': 'CaseID', 'studysubjectid': 'StudySubjectID'})
        dfs[4] = dfs[4].rename(columns={'site': 'SITE', 'caseid': 'CaseID', 'studysubjectid': 'StudySubjectID'})
                
        IDs = ['SITE', 'CaseID', 'StudySubjectID']
        feat_conscious = ['HxLOC', 'TotalGCSManual', 'TotalGCS', 'AVPUDetails']
        feat_pain = ['PtCompPainHead', 'PtCompPainFace', 'PtCompPainNeck', 'PtCompPainNeckMove', 'PtCompPainChest', 'PtCompPainBack', 'PtCompPainFlank', 'PtCompPainAbd', 'PtCompPainPelvis', 'PtCompPainExt']
        feat_tender = ['PtTenderHead', 'PtTenderFace', 'PtTenderNeck', 'PtTenderNeckLevel', 'PtTenderNeckLevelC1', 'PtTenderNeckLevelC2', 'PtTenderNeckLevelC3', 'PtTenderNeckLevelC4', 'PtTenderNeckLevelC5', 'PtTenderNeckLevelC6', 'PtTenderNeckLevelC7', 'PtTenderNeckAnt', 'PtTenderNeckPos', 'PtTenderNeckLat', 'PtTenderNeckMid', 'PtTenderNeckOther', 'PtTenderChest', 'PtTenderBack', 'PtTenderFlank', 'PtTenderAbd', 'PtTenderPelvis', 'PtTenderExt']
        feat_torticollis = ['LimitedRangeMotion']
        feat_otherinjury = ['OtherInjuries', 'OtherInjuriesHead', 'OtherInjuriesFace', 'OtherInjuriesNeck', 'OtherInjuriesChest', 'OtherInjuriesBack', 'OtherInjuriesFlank', 'OtherInjuriesAbd', 'OtherInjuriesPelvis', 'OtherInjuriesExt', 'MinorInjuries', 'MinorInjuriesHead', 'MinorInjuriesFace', 'MinorInjuriesNeck', 'MinorInjuriesChest', 'MinorInjuriesBack', 'MinorInjuriesFlank', 'MinorInjuriesAbs', 'MinorInjuriesPelv', 'MinorInjuriesExt']
        feat_neuro = ['PtParesthesias', 'PtSensoryLoss', 'PtExtremityWeakness', 'OtherNeuroDeficitDescCat']
        feat_ambulatory = ['PtAmbulatoryPriorArrival']
        feat_intervention = ['CervicalSpineImmobilization', 'CervicalSpineIntervCC', 'CervicalSpineIntervRLB', 'CervicalSpineIntervOther', 'MedsRecdPriorArrival', 'MedsRecdAna', 'MedsRecdGlu', 'MedsRecdPar', 'MedsRecdSed', 'MedsRecdOR', 'ArrPtIntub']
        feat_suspicious = ['DxCspineInjury']
        
        feat_demog = ['AgeInYears', 'Gender']
        
        feat_medhistory = ['BodyAsAWhole', 'Genitourinary1', 'Endocrinological1', 'Endocrinological2', 'HematologicLymphatic1', 'HematologicLymphatic2', 'HematologicLymphatic3', 'Neurological', 'Musculoskeletal']
        
        feat_injurymech = ['InjuryPrimaryMechanism', 'clotheslining', 'HeadFirst', 'HeadFirstRegion']
        
        df_merged = functools.reduce(lambda  left,right: pd.merge(left, right, on=IDs, how='left'), [dfs[3][IDs + ['ControlType'] + feat_conscious + feat_pain + feat_tender + feat_torticollis + feat_otherinjury + feat_neuro + feat_ambulatory + feat_intervention], dfs[4][IDs + feat_demog], dfs[6][IDs + feat_injurymech], dfs[8][IDs + feat_medhistory], dfs[0].drop(columns = ['ControlType'])])

        return df_merged

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # All to numeric
        df = cleaned_data.replace(['Y', 'YES', 'A'], 1)       
        df = df.replace(['N', 'NO'], 0)
        df = df.replace(['ND', 'NA'], float("NaN"))
        
                                                                                            
        # drop cols with vals missing this percent of the time
        # df = cleaned_data.dropna(axis=1, thresh=(1 - kwargs['frac_missing_allowed']) * cleaned_data.shape[0])

        # impute missing values
        # fill in values for some vars from unknown -> None
        # df = cleaned_data.dropna(axis=0)

        # Impute: Use domain knowledge
        liberal_feats = ['FocalNeuroFindings', 'FocalNeuroFindings2', 'Torticollis', 'Torticollis2', 
                          'SubInj_Head', 'SubInj_Face', 'SubInj_Ext', 'SubInj_TorsoTrunk', 'subinj_Head2', 'subinj_Face2', 'subinj_Ext2', 'subinj_TorsoTrunk2',
                          'Predisposed', 
                          'HighriskMVC', 'HighriskDiving', 'HighriskFall', 'HighriskHanging', 'HighriskHitByCar', 'HighriskOtherMV', 'AxialLoadAnyDoc', 'axialloadtop', 'Clotheslining']
        conserv_feats = ['LOC']
        unclear_feats = ['AlteredMentalStatus', 'AlteredMentalStatus2', 'ambulatory', 'PainNeck', 'PainNeck2', 'PosMidNeckTenderness', 'PosMidNeckTenderness2', 'TenderNeck', 'TenderNeck2']
        df[liberal_feats] = df[liberal_feats].fillna(0)
        df[conserv_feats] = df[conserv_feats].fillna(1)
        df[unclear_feats] = df[unclear_feats].fillna(0) 
        
        # pandas impute missing values with median
        df = df.fillna(df.median())
        # df.GCSScore = df.GCSScore.fillna(df.GCSScore.median())
        
        # df = df[df['ControlType'].isin(['case', 'ems'])]

        df.loc[:, 'outcome'] = (df['ControlType'] == 'case').astype(int)

        return df

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # add engineered featuures
        # df = helper.derived_feats(preprocessed_data)
        df = preprocessed_data
        # convert feats to dummy

        # AnalysisVariable features
        feat_av = ['AlteredMentalStatus', 'LOC', 'ambulatory', 'FocalNeuroFindings',
       'PainNeck', 'PosMidNeckTenderness', 'TenderNeck', 'Torticollis',
       'SubInj_Head', 'SubInj_Face', 'SubInj_Ext', 'SubInj_TorsoTrunk',
       'Predisposed', 'HighriskDiving', 'HighriskFall', 'HighriskHanging',
       'HighriskHitByCar', 'HighriskMVC', 'HighriskOtherMV', 'AxialLoadAnyDoc',
       'axialloadtop', 'Clotheslining', 'AlteredMentalStatus2',
       'FocalNeuroFindings2', 'PainNeck2', 'PosMidNeckTenderness2',
       'TenderNeck2', 'Torticollis2', 'subinj_Head2', 'subinj_Face2',
       'subinj_Ext2', 'subinj_TorsoTrunk2']
        
        df = pd.get_dummies(df, dummy_na=True, drop_first = True, columns = list(set(df.columns) - set(['SITE', 'CaseID', 'StudySubjectID', 'ControlType', 'AgeInYears', 'outcome'] + feat_av)))  # treat na as a separate category
        agegroup_df = pd.get_dummies(pd.cut(df['AgeInYears'], bins=[0, 2, 6, 12, 16],
                                   labels=['infant', 'preschool', 'school_age', 'adolescents'],
                                   include_lowest=True), prefix='age')
        df = pd.concat([df.drop(columns = 'AgeInYears'), agegroup_df], axis = 1)

        # remove any col that is all 0s
        df = df.loc[:, (df != 0).any(axis=0)]
        
        # remove the _no columns
        # if kwargs['drop_negative_columns']:
         #    df.drop([k for k in df.keys() if k.endswith('_no')], inplace=True)

        # remove site, case ID, subject ID, control type
        feats = df.keys()[4:]
        feats = feats.append(pd.Index(['SITE']))

        print(df.shape)
        
        return df[feats]

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
        #     split = tuple(preprocessed_data[preprocessed_data['SITE'].isin(site_split[0])],
        #                  preprocessed_data[preprocessed_data['SITE'].isin(site_split[1])],
        #                  preprocessed_data[preprocessed_data['SITE'].isin(site_split[2])])
        # elif kwargs['data_split'] == 'random_split':
        #     split = tuple(np.split(
        #     preprocessed_data.sample(frac=1, random_state=42),
        #     [int(.6 * len(preprocessed_data)),  # 60% train
        #      int(.8 * len(preprocessed_data))]  # 20% tune, 20% test
        # ))
        
        sites = np.arange(1, 18)
        np.random.seed(42)
        np.random.shuffle(sites)
        site_split = np.split(sites, [4])
        split = tuple(np.split(
            preprocessed_data[preprocessed_data['SITE'].isin(site_split[1])].sample(frac=1, random_state=42), 
            [int(.75 * sum(preprocessed_data['SITE'].isin(site_split[1])))]) +  # 60% train 20% tune
                      [preprocessed_data[preprocessed_data['SITE'].isin(site_split[0])]]) # 20% test
        return split
                                                                                            
    def get_outcome_name(self) -> str:
        return 'csi'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'csi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        return ['SITE', 'CaseID']  # keys which are useful but not used for prediction

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        return {
            'clean_data': {},
            'preprocess_data': {
                # drop cols with vals missing this percent of the time
                'frac_missing_allowed': [0.05, 0.10],
            },
            'extract_features': {
                # whether to drop columns with suffix _no
                'drop_negative_columns': [False],  # default value comes first
            },
        }


if __name__ == '__main__':
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=True)
    print('successfuly processed data\nshapes:',
          df_train.shape, df_tune.shape, df_test.shape,
          '\nfeatures:', list(df_train.columns))
