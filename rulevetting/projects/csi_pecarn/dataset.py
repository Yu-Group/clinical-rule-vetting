from os.path import join as oj

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict

import rulevetting
import rulevetting.api.util
from rulevetting.projects.csi_pecarn import helper
from rulevetting.templates.dataset_new import DatasetTemplate

class Dataset(DatasetTemplate):
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        # all the fnames to be loaded and searched over
        fnames = sorted([
            fname for fname in os.listdir(raw_data_path)
            if 'csv' in fname
            ])

        # read through each fname and save into the r dictionary
        r = {}
        print('read all the csvs...', fnames)
        if len(fnames) == 0:
            print('no csvs found in path', raw_data_path)
        for fname in tqdm(fnames):
            df = pd.read_csv(oj(raw_data_path, fname), encoding="ISO-8859-1")
            df.rename(columns={'site': 'SiteID'}, inplace=True)
            df.rename(columns={'SITE': 'SiteID'}, inplace=True)
            df.rename(columns={'caseid': 'CaseID'}, inplace=True)
            df.rename(columns={'controltype': 'ControlType'}, inplace=True)
            df.rename(columns={'studysubjectid': 'SubjectID'}, inplace=True)
            df.rename(columns={'StudySubjectID': 'SubjectID'}, inplace=True)
            assert ('SiteID' in df.keys())
            assert ('CaseID' in df.keys())
            assert ('ControlType' in df.keys())
            assert ('SubjectID' in df.keys())
            r[fname] = df

        # loop over the relevant forms and merge into one big df
        fnames_small = [fname for fname in fnames
                        if not 'radiology' in fname
                            and not 'kappa' in fname
                            and not 'injuryclassification' in fname
                            and not 'outside' in fname
                            and not 'onfield' in fname
                            and not 'medicalhistory' in fname]

        df_features = r[fnames_small[0]]
        print('merge all the dfs...', fnames_small)
        for i, fname in tqdm(enumerate(fnames_small)):
            df2 = r[fname].copy()
            
            # if subj has multiple entries, only keep first
            df2 = df2.drop_duplicates(subset=['SubjectID'], keep='last')
            
            # don't save duplicate columns
            df_features = df_features.set_index('SubjectID').combine_first(df2.set_index('SubjectID')).reset_index()
        
        # -----
        var_1 = ['AVPUDetails', 'AgeInYears', 'AlteredMentalStatus', 'ArrPtIntub', 'Assault',
                 'AxialLoadAnyDoc', 'CaseID', 'CervicalSpineImmobilization', 'ChildAbuse', 'ControlType',
                 'DxCspineInjury', 'Ethnicity', 'FallDownStairs', 'FallFromElevation', 'FocalNeuroFindings',
                 'Gender', 'HeadFirst', 'HighriskDiving', 'HighriskFall', 'HighriskHanging',
                 'HighriskHitByCar', 'HighriskMVC', 'HighriskOtherMV', 'InjuryPrimaryMechanism', 'IntervForCervicalStab',
                 'LOC', 'LimitedRangeMotion','LongTermRehab', 'Predisposed', "MedsGiven",
                 "MedsRecdPriorArrival", "MotorGCS", "PainNeck", "PainNeck2", "PassRestraint",
                 "PosMidNeckTenderness", 'PosMidNeckTenderness2',"PtAmbulatoryPriorArrival", "PtCompPain"] # 39
        
        var_2 = ['PtCompPainHead', 'PtCompPainNeck', 'PtCompPainPelvis', 'PtExtremityWeakness', 'PtParesthesias',
                 'PtSensoryLoss', 'PtTender', 'PtTenderAbd', 'PtTenderBack', 'PtTenderChest',
                 'PtTenderExt', 'PtTenderFlank', 'PtTenderHead', 'PtTenderNeck', 'PtTenderPelvis',
                 'ShakenBabySyndrome', 'SiteID', 'SubInj_Ext', 'SubInj_Face', 'SubInj_Head',
                 'SubInj_TorsoTrunk', 'SubjectID', 'TenderNeck', 'Torticollis', 'TotalGCS',
                 'ambulatory', 'axialloadtop', 'helmet'] # 28
        
        var_use = var_1 + var_2
        
        df = df_features[var_use]        
        df = df.drop(columns=['PosMidNeckTenderness2','PainNeck2','PtAmbulatoryPriorArrival','HeadFirst','PtCompPainNeck',
                              'PtTenderNeck','TotalGCS'])
        df = helper.rename_values(df)  
        
        return df

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        df = cleaned_data
    
        ### code for only using random control
        '''
        ind1=[True for i in range(cleaned_data.shape[0])]
        index=pd.array(ind1,dtype='boolean')
        for i in range(len(index)):
            if df.ControlType[i]=='case' or  df.ControlType[i]=='ran':
                index[i]=True
            else:
                index[i]=False
        df=df[index]
        '''
        
        #df=df[(df.ControlType == 'case') or (df.ControlType == 'ran')]
        df = df.assign(outcome=lambda x: (x.ControlType == 'case').astype(int))
        
        # drop cols with vals missing this percent of the time
        df = df.dropna(axis=1, thresh=(1 - kwargs['frac_missing_allowed']) * df.shape[0])
        df = df.drop(columns=['AgeInYears','InjuryPrimaryMechanism'])
       
        # impute missing values with median
        df = df.fillna(df.median())  
        
        #df['outcome'] = df[self.get_outcome_name()]
        #df = df.assign(outcome=lambda x: (x.ControlType == 'case').astype(int))
        
        return df
        

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame: 
        df = preprocessed_data
        return df

    # def get_outcome_name(self) -> str:
    #    return 'csi_intervention'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'csi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        return ["SubjectID", "SiteID", "CaseID", "ControlType", "Ethnicity", "Gender"]
        # keys which are useful but not used for prediction
        
    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        return {
            'clean_data': {},
            'preprocess_data': {
                # drop cols with vals missing this percent of the time
                'frac_missing_allowed': [0.15, 0.05],
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
