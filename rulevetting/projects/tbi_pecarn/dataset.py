from os.path import join as oj

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict

import rulevetting
import rulevetting.api.util
# TODO ?
from rulevetting.projects.iai_pecarn import helper
from rulevetting.templates.dataset import DatasetTemplate


class Dataset(DatasetTemplate):
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:

        # raw data path
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        # raw data file names to be loaded and searched over
        # for tbi, we only have one file
        fnames = sorted([
            fname for fname in os.listdir(raw_data_path)
            if 'csv' in fname])

        # read raw data
        for fname in tqdm(fnames):
            df = pd.read_csv(oj(raw_data_path, fname))

        return df

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        tbi_df = cleaned_data.copy()

        ################################
        # Step 1: Remove variables which have nothing to do with our problem (uncontroversial choices)
        ################################

        list1 = ['EmplType', 'Certification']
        list2 = ['InjuryMech']

        # grab all of the CT/Ind variables
        list3 = []
        for col in tbi_df.keys():
            if 'Ind' in col or 'CT' in col:
                list3.append(col)

        list4 = ['AgeTwoPlus', 'AgeInMonth']

        # grab all of the Finding variables
        list5 = ['Observed', 'EDDisposition']

        for col in tbi_df.keys():
            if 'Finding' in col:
                list5.append(col)

        total_rem = list1 + list2 + list3 + list4 + list5

        tbi_df = tbi_df.drop(total_rem, axis=1)
        ################################
        # Step 2: Remove variables with really high missingness (that we don't care about)
        ################################

        tbi_df = tbi_df.drop(['Ethnicity', 'Dizzy'], axis=1)

        ################################
        # Step 3: Remove observations with GCS < 14
        ################################

        tbi_df = tbi_df[tbi_df['GCSGroup'] == 2]
        tbi_df = tbi_df.drop(['GCSGroup'], axis=1)

        ################################
        # Step 4: Remove Missing Observations Among the Response Outcomes
        ################################

        tbi_df = tbi_df.dropna(subset=['DeathTBI', 'Intub24Head', 'Neurosurgery', 'HospHead'])
        tbi_df['PosIntFinal'].fillna(0, inplace=True)

        ################################
        # Step 5: Make a New Column ciTBI, without the hospitalization condition
        ################################

        sset = tbi_df[['DeathTBI', 'Intub24Head', 'Neurosurgery']]
        new_outcome = np.zeros(len(sset))
        new_outcome[np.sum(np.array(sset), 1) > 0] = 1

        tbi_df = tbi_df.assign(PosIntFinalNoHosp=new_outcome)

        ################################
        # Step 6: GCS missing values
        ################################

        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSVerbal'].isnull(), 'GCSVerbal'] = 5
        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSMotor'].isnull(), 'GCSMotor'] = 6
        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSEye'].isnull(), 'GCSEye'] = 4

        tbi_df.drop(tbi_df[(tbi_df['GCSTotal'] == 15) & (
                (tbi_df['GCSVerbal'] < 5) | (tbi_df['GCSMotor'] < 6) | (
                tbi_df['GCSEye'] < 4))].index, inplace=True)

        tbi_df.drop(tbi_df[(tbi_df['GCSTotal'] == 14) & (
                (tbi_df['GCSVerbal'].isnull()) | (tbi_df['GCSMotor'].isnull()) | (
            tbi_df['GCSEye'].isnull()))].index, inplace=True)

        ################################
        # Step 7: Paralyzed, Sedated, Intubated missing values
        ################################

        tbi_df.drop(tbi_df.loc[(tbi_df['Paralyzed'] == 1) | (tbi_df['Sedated'] == 1) | (
                tbi_df['Intubated'] == 1)].index, inplace=True)
        tbi_df.drop(tbi_df.loc[(tbi_df['Paralyzed'].isnull()) | (tbi_df['Sedated'].isnull()) | (
            tbi_df['Intubated'].isnull())].index, inplace=True)
        tbi_df.drop(['Sedated', 'Paralyzed', 'Intubated'], axis=1, inplace=True)

        ################################
        # Step 8: AMS missing values
        ################################

        tbi_df.drop(tbi_df.loc[tbi_df['AMS'].isnull()].index, inplace=True)

        ################################
        # Step 9: OSI missing values
        ################################

        tbi_df.drop(tbi_df.loc[tbi_df['OSI'].isnull()].index, inplace=True)

        ################################
        # Step 10: Hemotoma related features missing values
        ################################


        tbi_df.drop(tbi_df.loc[(tbi_df['Hema'].isnull()) | (tbi_df['HemaLoc'].isnull()) | (
            tbi_df['HemaSize'].isnull())].index, inplace=True)

        ################################
        # Step 11: skull fracture related features missing values
        ################################

        tbi_df.loc[(tbi_df['SFxPalp'] == 2), 'SFxPalp'] = 1
        tbi_df.drop(tbi_df.loc[
                        (tbi_df['FontBulg'].isnull()) | (tbi_df['SFxPalpDepress'].isnull()) | (
                            tbi_df['SFxPalp'].isnull())].index, inplace=True)

        ################################
        # Step 12: skull fracture Bas. Removed missing values
        ################################

        tbi_df.drop(tbi_df.loc[tbi_df['SFxBas'].isnull()].index, inplace=True)

        ################################
        # Step 13: clavicle related features missing values
        ################################

        tbi_df.drop(tbi_df.loc[tbi_df['Clav'].isnull()].index, inplace=True)

        ################################
        # Step 14: neurological injury related features missing values
        ################################

        tbi_df.drop(tbi_df.loc[tbi_df['NeuroD'].isnull()].index, inplace=True)

        ################################
        # Step 15: vomiting related features missing values
        ################################

        tbi_df.drop(['VomitStart', 'VomitLast', 'VomitNbr'], axis=1, inplace=True)
        tbi_df.drop(tbi_df.loc[tbi_df['Vomit'].isnull()].index, inplace=True)

        ################################
        # Step 16: Headache related features missing values
        ################################

        tbi_df.drop(['HAStart'], axis=1, inplace=True)
        tbi_df.drop(
            tbi_df.loc[(tbi_df['HA_verb'].isnull()) | (tbi_df['HASeverity'].isnull())].index,
            inplace=True)

        ################################
        # Step 17: Seizure related features missing values
        ################################

        tbi_df.drop(
            tbi_df.loc[(tbi_df['Seiz'].isnull()) | (tbi_df['SeizLen'].isnull())].index,
            inplace=True)
        tbi_df.drop('SeizOccur', axis=1, inplace=True)

        ################################
        # Step 18: Loss of Consciousness features features missing values
        ################################

        tbi_df.drop(
            tbi_df.loc[(tbi_df['LOCSeparate'].isnull()) | (tbi_df['LocLen'].isnull())].index,
            inplace=True)
        tbi_df.loc[(tbi_df['LOCSeparate'] == 2), 'LOCSeparate'] = 1

        ################################
        # Step 19: Amnesia and High injury severity missing values
        ################################

        tbi_df.drop(tbi_df.loc[(tbi_df['Amnesia_verb'].isnull()) | (
            tbi_df['High_impact_InjSev'].isnull())].index, inplace=True)

        df = tbi_df.copy()
        df['outcome'] = df[self.get_outcome_name()]

        return df

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # put PatNum to the index
        df = preprocessed_data.copy()
        df.index = df.PatNum
        df = df.drop(['PatNum'], axis=1)

        # choose features which is not binary
        for col in df:
            if (col != 'AgeinYears') & (len(df[col].unique()) > 2):
                df[col] = df[col].astype('category')

        # convert these feats to dummy
        df = pd.get_dummies(df, dummy_na=True)  # treat na as a separate category

        # remove any col that is all 0s
        df = df.loc[:, (df != 0).any(axis=0)]

        return df

    def get_outcome_name(self) -> str:
        return 'PosIntFinal'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'tbi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        # TODO
        return []  # keys which are useful but not used for prediction

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        # TODO
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

    #NOTE: for quick reference - this is what's inherited and gets run:

    # def get_data(self, save_csvs: bool = False,
    #              data_path: str = rulevetting.DATA_PATH,
    #              load_csvs: bool = False,
    #              run_perturbations: bool = False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    #     """Runs all the processing and returns the data.
    #     This method does not need to be overriden.

    #     Params
    #     ------
    #     save_csvs: bool, optional
    #         Whether to save csv files of the processed data
    #     data_path: str, optional
    #         Path to all data
    #     load_csvs: bool, optional
    #         Whether to skip all processing and load data directly from csvs
    #     run_perturbations: bool, optional
    #         Whether to run / save data pipeline for all combinations of judgement calls

    #     Returns
    #     -------
    #     df_train
    #     df_tune
    #     df_test
    #     """
if __name__ == '__main__':
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=False)
    print('successfuly processed data\nshapes:',
          df_train.shape, df_tune.shape, df_test.shape,
          '\nfeatures:', list(df_train.columns))
