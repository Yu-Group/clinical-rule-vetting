import inspect
import os
from os.path import join as oj
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# TODO: fix _init_.py so these are easily accessible
import rulevetting.api.util
import rulevetting.projects.tbi_pecarn.helper as hp
from rulevetting.templates.dataset import DatasetTemplate


class Dataset(DatasetTemplate):

    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        """
        Convert the raw data files into a pandas dataframe.
        Dataframe keys should be reasonable (lowercase, underscore-separated).
        Data types should be reasonable.

        Params
        ------
        data_path: str, optional
            Path to all data files
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        cleaned_data: pd.DataFrame
        """

        # No judgement calls

        # raw data path
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        # raw data file names to be loaded and searched over
        # for tbi, we only have one file
        fnames = sorted([
            fname for fname in os.listdir(raw_data_path)
            if 'csv' in fname])

        # read raw data
        df = pd.DataFrame()
        for fname in tqdm(fnames):
            df = df.append(pd.read_csv(oj(raw_data_path, fname)))

        return df

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the data.
        Impute missing values.
        Scale/transform values.
        Should put the prediction target in a column named "outcome"

        Parameters
        ----------
        cleaned_data: pd.DataFrame
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        preprocessed_data: pd.DataFrame
        """

        tbi_df = cleaned_data.copy()

        if kwargs:
            judg_calls = kwargs["preprocess_data"]
        else:
            judg_calls = self.get_judgement_calls_current()

        ################################
        # Step 1: Remove variables which have nothing to do with our problem (uncontroversial choices)
        ################################

        list1 = ['EmplType', 'Certification']

        # judgement call: drop injury mechanic
        if not judg_calls["step1_injMech"]:
            list1.append('InjuryMech')

        # grab all of the CT/Ind variables
        list2 = []
        for col in tbi_df.keys():
            if 'Ind' in col or 'CT' in col:
                list2.append(col)

        list3 = ['AgeTwoPlus', 'AgeInMonth']

        # grab all of the Finding variables
        list4 = ['Observed', 'EDDisposition']

        for col in tbi_df.keys():
            if 'Finding' in col:
                list4.append(col)

        total_rem = list1 + list2 + list3 + list4

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
        # Step 4: Generate an unified response variable
        ################################
        # NOTE: PosIntFinalNoHosp is too wordy IMO, just call it ciTBI
        tbi_df = hp.union_var(tbi_df, ['DeathTBI', 'Intub24Head', 'Neurosurgery',
                                       'HospHead', 'PosIntFinal'], "ciTBI")

        ################################
        # Step 5: Impute/drop GCS Verbal/Motor/Eye Scores
        ################################

        # judgement call: drop borderline GCS scores with missing components
        if judg_calls["step5_missSubGCS"]:
            tbi_df.drop(tbi_df[(tbi_df['GCSTotal'] == 14) & (
                    (tbi_df['GCSVerbal'].isnull()) | (tbi_df['GCSMotor'].isnull()) | (
                tbi_df['GCSEye'].isnull()))].index, inplace=True)

        # Impute the missing values among GCS = 15 scores to just be the full points
        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSVerbal'].isnull(), 'GCSVerbal'] = 5
        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSMotor'].isnull(), 'GCSMotor'] = 6
        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSEye'].isnull(), 'GCSEye'] = 4

        # judgement call: Maximum total GCS but not the sum of subcomponents
        if judg_calls["step5_fake15GCS"]:
            tbi_df.drop(tbi_df[(tbi_df['GCSTotal'] == 15) & (
                    (tbi_df['GCSVerbal'] < 5) | (tbi_df['GCSMotor'] < 6) | (
                    tbi_df['GCSEye'] < 4))].index, inplace=True)

        # judgement call: Maximum subcomponents but not total:
        if judg_calls["step5_fake14GCS"]:
            tbi_df.drop(tbi_df[(tbi_df['GCSTotal'] == 14) &
                               (tbi_df['GCSVerbal'] == 5) &
                               (tbi_df['GCSMotor'] == 6) &
                               (tbi_df['GCSEye'] == 4)].index,
                        inplace=True)

        ################################
        # Step 6: Drop Paralyzed/Sedated/Intubated
        ################################
        # NOTE: is this a judgement call within our scope, or is it out of scope?
        # Right now: outside the scope for first-time CT evaluation

        # Drop the observations that were Intubated... and where the info is missing
        tbi_df.drop(tbi_df.loc[(tbi_df['Paralyzed'] == 1) | (tbi_df['Sedated'] == 1)
                               | (tbi_df['Intubated'] == 1)].index, inplace=True)
        tbi_df.drop(tbi_df.loc[(tbi_df['Paralyzed'].isnull()) | (tbi_df['Sedated'].isnull())
                               | (tbi_df['Intubated'].isnull())].index, inplace=True)

        # Drop these categories altogether
        tbi_df.drop(['Sedated', 'Paralyzed', 'Intubated'], axis=1, inplace=True)

        ################################
        # Step 7: Drop missing AMS
        ################################

        tbi_df.drop(tbi_df.loc[tbi_df['AMS'].isnull()].index, inplace=True)

        ################################
        # Step 8: Drop  those with missing OSI - other substantial injuries
        ################################

        if judg_calls["step8_missingOSI"]:
            tbi_df.drop(tbi_df.loc[tbi_df['OSI'].isnull()].index, inplace=True)

        ################################
        # Step 9: Impute/drop based on Hema variables
        ################################
        # TODO: UMBRELLA: there is a judgement call whether to flatten this or not, see below

        # Judgement call - impute HEMA from the presence of ANY sub-variable (Union),
        # drop sub-variables
        if judg_calls["step9_HEMA"] == 1:

            # fix so missings don't get counted as present
            tbi_df.loc[tbi_df[(tbi_df.HemaLoc == 92) | (tbi_df.HemaSize == 92)].index,
                       ['HemaLoc', 'HemaSize']] = np.NaN
            tbi_df = hp.union_var(tbi_df, ['Hema', 'HemaLoc', 'HemaSize'], 'Hema')

        # Judgement call: Permit missing size, impute from loc, drop size
        elif judg_calls["step9_HEMA"] == 2:
            # first drop really missing HemaLoc
            tbi_df.drop(tbi_df.loc[(tbi_df['HemaLoc'].isnull())].index, inplace=True)

            # impute Hema from HemaLoc
            tbi_df.loc[tbi_df[tbi_df.HemaLoc != 92].index, "Hema"] = 1

            # We don't care about the HemaSize
            tbi_df.drop(['HemaSize'], axis=1, inplace=True)

        # Judgement call: be strict about missing sub-categories
        elif judg_calls["step9_HEMA"] == 3:
            tbi_df.drop(tbi_df.loc[(tbi_df['HemaLoc'].isnull()) |
                                   (tbi_df['HemaSize'].isnull())].index,
                        inplace=True)
        else:
            raise NotImplementedError("Desired Hema preprocess step not implemented!")

        # now drop remaining missing Hema regardless of the above
        tbi_df.drop(tbi_df.loc[(tbi_df['Hema'].isnull())].index, inplace=True)

        ################################
        # Step 10: Impute/drop based on palpable skull fracture
        ################################
        # TODO: UMBRELLA: SFxPalpDepress

        # treat unclear tests of skull fracture as a sign of possible presence
        if judg_calls["step10_cautiousUncl"]:
            tbi_df.loc[(tbi_df['SFxPalp'] == 2), 'SFxPalp'] = 1

        # Fontanelle bulging is believed to be a good indicator so drop missing
        tbi_df.drop(tbi_df.loc[(tbi_df['SFxPalp'].isnull()) |
                               (tbi_df['FontBulg'].isnull()) |
                               (tbi_df['SFxPalpDepress'].isnull())].index,
                    inplace=True)

        ################################
        # Step 11: Impute/drop based on basilar skull fracture variables
        ################################
        # TODO: UMBRELLA: SFxBasHem SFxBasOto SFxBasPer SFxBasRet SFxBasRhi

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[tbi_df['SFxBas'].isnull()].index, inplace=True)

        ################################
        # Step 12: Impute/drop based on Clav group of variables
        ################################
        # TODO: UMBRELLA: ClavFace ClavNeck ClavFro ClavOcc ClavPar ClavTem

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[tbi_df['Clav'].isnull()].index, inplace=True)

        ################################
        # Step 13: Impute/drop based on Neuro group of variables
        ################################
        # TODO: UMBRELLA: NeuroDMotor NeuroDSensory NeuroDCranial NeuroDReflex NeuroDOth

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[tbi_df['NeuroD'].isnull()].index, inplace=True)

        ################################
        # Step 14: Impute/drop based on Vomiting group of variables
        ################################

        if not judg_calls["step14_vomitDtls"]:
            # TODO: UMBRELLA (if present): 'VomitStart', 'VomitLast', 'VomitNbr
            tbi_df.drop(['VomitStart', 'VomitLast', 'VomitNbr'], axis=1, inplace=True)

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[tbi_df['Vomit'].isnull()].index, inplace=True)

        ################################
        # Step 15: Impute/drop based on Headache group of variables
        ################################
        # TODO: UMBRELLA: HASeverity HAStart
        # NOTE: (Andrej) This seems to me similar to HEMA, just maybe use a different default

        idx_verbal = (tbi_df.HA_verb != 91).index

        # Judgement call - impute HA_verb from the presence of ANY sub-variable (Union),
        # drop sub-variables
        if judg_calls["step15_HA"] == 1:

            # fix so missings don't get counted as present
            tbi_df.loc[tbi_df[(tbi_df.HAStart == 92) | (tbi_df.HASeverity == 92)].index,
                       ['HAStart', 'HASeverity']] = np.NaN

            # need to modify union_variable code due to 91 - preverbal
            tbi_df.loc[idx_verbal, 'HA_verb'] = tbi_df.loc[idx_verbal,
                                                           ["HA_verb", "HAStart", "HASeverity"]]. \
                any(axis=1, skipna=True).astype(int)

            # drop sub-features
            tbi_df.drop(["HAStart", "HASeverity"], axis=1, inplace=True)


        # Judgement call: Permit missing start, impute from severity, drop start
        elif judg_calls["step15_HA"] == 2:
            # first drop really missing HASeverity
            tbi_df.drop(tbi_df.loc[(tbi_df['HASeverity'].isnull())].index, inplace=True)

            # impute HA_verb from HASeverity
            tbi_df.loc[idx_verbal.intersection(tbi_df[tbi_df.HASeverity != 92].index),
                       "HA_verb"] = 1

            # We don't care about the HAStart
            tbi_df.drop(['HAStart'], axis=1, inplace=True)

        # Judgement call: be strict about missing sub-categories
        elif judg_calls["step15_HA"] == 3:

            # Judgement call: keep HAStart
            if not judg_calls["step15_HAStart"]:
                tbi_df.drop(['HAStart'], axis=1, inplace=True)

            else:
                tbi_df.drop(tbi_df.loc[(tbi_df['HAStart'].isnull())].index,
                            inplace=True)

            tbi_df.drop(tbi_df.loc[(tbi_df['HA_verb'].isnull()) |
                                   (tbi_df['HASeverity'].isnull())].index,
                        inplace=True)

        ################################
        # Step 16: Impute/drop based on Seizure group of variables
        ################################
        # TODO: UMBRELLA: SeizOccur SeizLen
        # NOTE: (Andrej) This seems to me similar to HEMA, just maybe use a different default

        # Judgement call: union the Seiz var with sub-vars
        if judg_calls["step16_Seiz"] == 1:

            # fix so missings don't get counted as present
            tbi_df.loc[tbi_df[(tbi_df.SeizLen == 92) | (tbi_df.SeizOccur == 92)].index,
                       ['SeizLen', 'SeizOccur']] = np.NaN

            tbi_df = hp.union_var(tbi_df, ['Seiz', 'SeizLen', 'SeizOccur'], 'Seiz')


        # Judgement call: Permit missing SeizOccur, impute from SeizLen, drop SeizOccur
        elif judg_calls["step16_Seiz"] == 2:
            # first drop really missing SeizLen
            tbi_df.drop(tbi_df.loc[(tbi_df['SeizLen'].isnull())].index, inplace=True)

            # impute Seiz from SeizLen
            tbi_df.loc[tbi_df[tbi_df.SeizLen != 92].index, "Seiz"] = 1

            # We don't care about the SeizOccur
            tbi_df.drop(['SeizOccur'], axis=1, inplace=True)

        # Judgement call: be strict about missing sub-categories
        elif judg_calls["step16_Seiz"] == 3:

            # Judgement call: keep SeizOccur
            if not judg_calls["step16_SeizOccur"]:
                tbi_df.drop(['SeizOccur'], axis=1, inplace=True)

            else:
                tbi_df.drop(tbi_df.loc[(tbi_df['SeizOccur'].isnull())].index,
                            inplace=True)

            tbi_df.drop(tbi_df.loc[(tbi_df['Seiz'].isnull()) |
                                   (tbi_df['SeizLen'].isnull())].index,
                        inplace=True)

        ################################
        # Step 17: Impute/drop based on Loss of Consciousness variables
        ################################
        if judg_calls["step17_cautiousUncl"]:
            tbi_df.loc[(tbi_df['LOCSeparate'] == 2), 'LOCSeparate'] = 1

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[(tbi_df['LOCSeparate'].isnull()) |
                               (tbi_df['LocLen'].isnull())].index,
                    inplace=True)

        ################################
        # Step 18: Drop Missing Values for Amnesia/High Injury Severity
        ################################

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[(tbi_df['Amnesia_verb'].isnull()) |
                               (tbi_df['High_impact_InjSev'].isnull())].index,
                    inplace=True)

        ################################
        # Step 19: Drop the Drugs Column
        ################################
        if not judg_calls["step19_Drugs"]:
            tbi_df = tbi_df.drop('Drugs', axis=1)

        tbi_df['outcome'] = tbi_df[self.get_outcome_name()]

        return tbi_df

    # TODO: - check
    #       - binarization of categoricals !!!
    #       - flattening of umbrellas !!!
    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """
        Binarizes the categoricals
        Flattens depending on the


        Parameters
        ----------
        preprocessed_data: pd.DataFrame
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        extracted_features: pd.DataFrame
        """

        # TODO: implement consistent binarization
        judg_calls = self.get_judgement_calls_current()

        # put PatNum to the index
        # FIXME: check index vs actual length
        df = preprocessed_data.copy()
        df.index = df.PatNum
        df = df.drop(['PatNum'], axis=1)

        # choose features which is not binary
        for col in df:
            if (col != 'AgeinYears') & (len(df[col].unique()) > 2):
                df[col] = df[col].astype('category')

        # convert these feats to dummy
        df = pd.get_dummies(df, dummy_na=True)  # treat na as a separate category

        # remove any col that is all 0s - constant value in all observations
        df = df.loc[:, (df != 0).any(axis=0)]

        return df

    def get_outcome_name(self) -> str:
        return 'PosIntFinal'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'tbi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        # TODO gender race
        return []  # keys which are useful but not used for prediction

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        """
        Returns a dictionary of keyword arguments for each function in the dataset class.
        Each key should be a string with the name of the arg.
        Each value should be a list of values, with the default value coming first.

        Example
        -------
        return {
            'clean_data': {},
            'preprocess_data': {
                'imputation_strategy': ['mean', 'median'],  # first value is default
            },
            'extract_features': {},
        }
        """

        # TODO: document in-place
        judg_calls = \
            {
                'clean_data'      : {},
                'preprocess_data' : {
                    # drop cols with vals missing this percent of the time
                    # 'frac_missing_allowed': [0.05, 0.10],

                    # include injury mechanic
                    "step1_injMech"      : [False, True],
                    "step5_missSubGCS"   : [True, False],
                    "step5_fake15GCS"    : [True, False],
                    "step5_fake14GCS"    : [True, False],
                    "step8_missingOSI"   : [True, False],
                    "step9_HEMA"         : [3, 1, 2],
                    "step10_cautiousUncl": [True, False],
                    "step14_vomitDtls"   : [False, True],
                    "step15_HA"          : [2, 3, 1],
                    # only affects 3 above
                    "step15_HAStart"     : [False, True],
                    "step16_Seiz"        : [2, 3, 1],
                    # only affects 3 above
                    "step16_SeizOccur"   : [False, True],
                    "step17_cautiousUncl": [True, False],
                    "step19_Drugs"       : [False, True]

                },
                'extract_features': {
                    # whether to drop columns with suffix _no
                    # 'drop_negative_columns': [False],  # default value comes first
                },
            }

        return judg_calls

    def get_judgement_calls_current(self) -> Dict[str, list]:
        """
         Returns the sub-dictionary of judgement calls for the calling function
         with default values, using inspection.

        Returns
        -------
        Dict[str, list]
            DESCRIPTION.

        """

        calling_func = inspect.currentframe().f_back.f_code.co_name
        judg_calls = self.get_judgement_calls_dictionary()[calling_func]

        # return the default
        return {k: v[0] for k, v in judg_calls.items()}

    # NOTE: for quick reference - this is what's inherited and gets run:
    # NOTE: can actually override it if extra judgement call functionality needed!

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
    # for development
    self = Dataset()
    raw_data_path = oj(rulevetting.DATA_PATH, self.get_dataset_id(), 'raw')

    # raw data file names to be loaded and searched over
    # for tbi, we only have one file
    fnames = sorted([
        fname for fname in os.listdir(raw_data_path)
        if 'csv' in fname])

    # read raw data
    cleaned_data = pd.DataFrame()
    for fname in tqdm(fnames):
        cleaned_data = cleaned_data.append(pd.read_csv(oj(raw_data_path, fname)))

    # df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=False)

    # dset.preprocess_data(df_train)

    # print('successfuly processed data\nshapes:',
    #       df_train.shape, df_tune.shape, df_test.shape,
    #       '\nfeatures:', list(df_train.columns))
