import inspect
import os
import random
from os.path import join as oj
from typing import Dict

import numpy as np
import pandas as pd
from joblib import Memory
from tqdm import tqdm
from vflow import init_args, Vset, build_Vset

# TODO: fix _init_.py so these are easily accessible
import rulevetting
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
        # Step 1: Remove variables which have nothing to do with our problem
        # (uncontroversial choices, these variables do not matter for our problem at all)
        ################################

        list1 = ['EmplType', 'Certification']

        # judgement call: drop injury mechanic
        if not judg_calls["step1_injMech"]:
            list1.append('InjuryMech')

        # grab all of the CT/Ind variables, which have to do with CT scans
        # when our classifier is applied, we will not have access to this info
        list2 = []
        for col in tbi_df.keys():
            if 'Ind' in col or 'CT' in col:
                list2.append(col)

        # 'AgeTwoPlus' can be recreated easily, 'AgeInMonth' we decided does not matter
        list3 = ['AgeTwoPlus', 'AgeInMonth']

        # These vars have to do with obtaining CT scans or reasons for hospital discharge
        # These vars will not be observed in our case
        list4 = ['Observed', 'EDDisposition']

        # these vars are all indicators for what was found on a CT scan (not relevant for us)
        for col in tbi_df.keys():
            if 'Finding' in col:
                list4.append(col)

        # combine all lists and drop
        total_rem = list1 + list2 + list3 + list4

        tbi_df = tbi_df.drop(total_rem, axis=1)

        ################################
        # Step 2: Remove variables with really high missingness
        # Dizzy is unimportant/too subjective according to Dr. Inglis
        # Ethnicity is too missing, so difficult to perform meaningful posthoc analysis
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

        tbi_df = hp.union_var(tbi_df, ['DeathTBI', 'Intub24Head', 'Neurosurgery',
                                       'HospHead', 'PosIntFinal'], "outcome")

        ################################
        # Step 5: Impute/drop GCS Verbal/Motor/Eye Scores
        ################################

        # judgement call: drop borderline GCS scores with missing components
        if judg_calls["step5_missSubGCS"]:
            tbi_df.drop(tbi_df[(tbi_df['GCSTotal'] == 14) &
                               ((tbi_df['GCSVerbal'].isnull()) |
                                (tbi_df['GCSMotor'].isnull()) |
                                (tbi_df['GCSEye'].isnull()))].index,
                        inplace=True)

        # Impute the missing values among GCS = 15 scores to just be the full points
        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSVerbal'].isnull(), 'GCSVerbal'] = 5
        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSMotor'].isnull(), 'GCSMotor'] = 6
        tbi_df.loc[(tbi_df['GCSTotal'] == 15) & tbi_df['GCSEye'].isnull(), 'GCSEye'] = 4

        # judgement call: drop obs who have max total GCS but not max's of subcomponents
        if judg_calls["step5_fake15GCS"]:
            tbi_df.drop(tbi_df[(tbi_df['GCSTotal'] == 15) &
                               ((tbi_df['GCSVerbal'] < 5) |
                                (tbi_df['GCSMotor'] < 6) |
                                (tbi_df['GCSEye'] < 4))].index,
                        inplace=True)

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

        # Drop the observations that were Intubated... and where the info is missing
        tbi_df.drop(tbi_df.loc[(tbi_df['Paralyzed'] == 1) | (tbi_df['Sedated'] == 1)
                               | (tbi_df['Intubated'] == 1)].index, inplace=True)
        tbi_df.drop(tbi_df.loc[(tbi_df['Paralyzed'].isnull()) | (tbi_df['Sedated'].isnull())
                               | (tbi_df['Intubated'].isnull())].index, inplace=True)

        # Drop these features altogether
        tbi_df.drop(['Sedated', 'Paralyzed', 'Intubated'], axis=1, inplace=True)

        ################################
        # Step 7: Drop missing AMS
        ################################

        tbi_df.drop(tbi_df.loc[tbi_df['AMS'].isnull()].index, inplace=True)

        ################################
        # Step 8: Drop those with missing OSI - other substantial injuries
        ################################

        if judg_calls["step8_missingOSI"]:
            tbi_df.drop(tbi_df.loc[tbi_df['OSI'].isnull()].index, inplace=True)
        # don't see any alternative than to drop the whole thing, since where
        # could this be even imputed from
        else:
            tbi_df.drop(['OSI'], axis=1, inplace=True)

        ################################
        # Step 9: Impute/drop based on Hema variables
        ################################
        # UMBRELLA: there is a judgement call whether to flatten this or not, see below

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
        # UMBRELLA: SFxPalpDepress - doesn't work with cautious because
        # unclear gets mapped to 92

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
        # UMBRELLA: SFxBasHem SFxBasOto SFxBasPer SFxBasRet SFxBasRhi

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[tbi_df['SFxBas'].isnull()].index, inplace=True)

        ################################
        # Step 12: Impute/drop based on Clav group of variables
        ################################
        # UMBRELLA: ClavFace ClavNeck ClavFro ClavOcc ClavPar ClavTem

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[tbi_df['Clav'].isnull()].index, inplace=True)

        ################################
        # Step 13: Impute/drop based on Neuro group of variables
        ################################
        # UMBRELLA: NeuroDMotor NeuroDSensory NeuroDCranial NeuroDReflex NeuroDOth

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[tbi_df['NeuroD'].isnull()].index, inplace=True)

        ################################
        # Step 14: Impute/drop based on Vomiting group of variables
        ################################

        if not judg_calls["step14_vomitDtls"]:
            #  UMBRELLA (if present): 'VomitStart', 'VomitLast', 'VomitNbr
            tbi_df.drop(['VomitStart', 'VomitLast', 'VomitNbr'], axis=1, inplace=True)

        # Not possible to impute, just drop missing
        tbi_df.drop(tbi_df.loc[tbi_df['Vomit'].isnull()].index, inplace=True)

        ################################
        # Step 15: Impute/drop based on Headache group of variables
        ################################
        #  UMBRELLA: HASeverity HAStart
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

            tbi_df.drop(tbi_df.loc[(tbi_df['HASeverity'].isnull())].index,
                        inplace=True)

        # now drop remaining missing HA_verb regardless of the above
        tbi_df.drop(tbi_df.loc[(tbi_df['HA_verb'].isnull())].index, inplace=True)

        ################################
        # Step 16: Impute/drop based on Seizure group of variables
        ################################
        # UMBRELLA: SeizOccur SeizLen
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

            tbi_df.drop(tbi_df.loc[(tbi_df['SeizLen'].isnull())].index,
                        inplace=True)

        # now drop remaining missing Seiz regardless of the above
        tbi_df.drop(tbi_df.loc[(tbi_df['Seiz'].isnull())].index, inplace=True)

        ################################
        # Step 17: Impute/drop based on Loss of Consciousness variables
        ###############################
        # UMBRELLA LocLen

        # Judgement call: unclear counts as present
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
            tbi_df.drop('Drugs', axis=1, inplace=True)
        else:
            tbi_df.drop(tbi_df.loc[tbi_df['Drugs'].isnull()].index,
                        inplace=True)

        ################################
        # Step 20: Handle the ActNorm column
        ################################
        # Judgement call: N/A counts as normal
        if judg_calls["step20_ActNormal"]:
            tbi_df.loc[tbi_df.ActNorm.isnull(), 'ActNorm'] = 1
        else:
            # N/A counts as ABnormal
            tbi_df.loc[tbi_df.ActNorm.isnull(), 'ActNorm'] = 0

        tbi_df['outcome'] = tbi_df[self.get_outcome_name()]
        tbi_df.set_index('PatNum', inplace=True)

        assert np.sum(np.sum(pd.isna(tbi_df.drop(self.get_meta_keys(), axis=1)))) == 0, \
            "N/As present after cleaning!"

        return tbi_df

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """
        Binarizes the categoricals
        Flattens depending on the the judgmement calls provided

        Parameters
        ----------
        preprocessed_data: pd.DataFrame
        kwargs: dict
            Dictionary of hyperparameters specifying judgement calls

        Returns
        -------
        extracted_features: pd.DataFrame
        """

        if kwargs:
            judg_calls = kwargs["extract_features"]
            prepr_calls = kwargs["preprocess_data"]
        else:
            # defaults
            judg_calls = self.get_judgement_calls_current()
            prepr_calls = self.get_judgement_calls_dictionary()["preprocess_data"]
            prepr_calls = {k: v[0] for k, v in prepr_calls.items()}

        df = preprocessed_data.copy()

        # FLATTEN UMBRELLAS:
        if judg_calls["HEMA_umbrella"]:

            # cannot flatten if unionized
            assert prepr_calls["step9_HEMA"] != 1

            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("Hema", axis=1, inplace=True)
            # 92 is treated as 0
            try:
                df.loc[(df.HemaLoc == 92) | (df.HemaSize == 92),
                       ['HemaLoc', 'HemaSize']] = 0
            except AttributeError:
                df.loc[df.HemaLoc == 92, 'HemaLoc'] = 0

        if judg_calls["SFxPalp_umbrella"]:
            # cannot disambiguate if unclear had been assigned to 1
            assert not prepr_calls["step10_cautiousUncl"]

            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("SFxPalp", axis=1, inplace=True)
            # assign NEW category to 92 - no palpable at all
            df.loc[df.SFxPalpDepress == 92, 'SFxPalpDepress'] = 2

        if judg_calls["SFxBas_umbrella"]:
            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("SFxBas", axis=1, inplace=True)
            # 92 is treated as 0
            df.loc[(df.SFxBasHem == 92) | (df.SFxBasOto == 92) |
                   (df.SFxBasPer == 92) | (df.SFxBasRet == 92) |
                   (df.SFxBasRhi == 92), ["SFxBasHem", "SFxBasOto", "SFxBasPer",
                                          "SFxBasRet", "SFxBasRhi"]] = 0

        if judg_calls["Clav_umbrella"]:
            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("Clav", axis=1, inplace=True)
            # 92 is treated as 0
            df.loc[(df.ClavFace == 92) | (df.ClavNeck == 92) |
                   (df.ClavFro == 92) | (df.ClavOcc == 92) |
                   (df.ClavPar == 92) | (df.ClavTem == 92),
                   ["ClavFace", "ClavNeck", "ClavFro", "ClavOcc", "ClavPar",
                    "ClavTem"]] = 0

        if judg_calls["AMS_umbrella"]:
            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("AMS", axis=1, inplace=True)
            # 92 is treated as 0
            df.loc[(df.AMSAgitated == 92) | (df.AMSSleep == 92) |
                   (df.AMSSlow == 92) | (df.AMSRepeat == 92) |
                   (df.AMSOth == 92),
                   ["AMSAgitated", "AMSSleep", "AMSSlow", "AMSRepeat", "AMSOth"]] = 0

        if judg_calls["NeuroD_umbrella"]:
            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("NeuroD", axis=1, inplace=True)
            # 92 is treated as 0
            df.loc[(df.NeuroDMotor == 92) | (df.NeuroDSensory == 92) |
                   (df.NeuroDCranial == 92) | (df.NeuroDReflex == 92) |
                   (df.NeuroDOth == 92),
                   ["NeuroDMotor", "NeuroDSensory", "NeuroDCranial", "NeuroDReflex",
                    "NeuroDOth"]] = 0

        if judg_calls["Vomit_umbrella"]:
            # the child vars must not have been removed
            assert prepr_calls["step14_vomitDtls"]

            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("Vomit", axis=1, inplace=True)
            # 92 is treated as 0
            df.loc[(df.VomitStart == 92) | (df.VomitLast == 92) |
                   (df.VomitNbr == 92),
                   ["VomitStart", "VomitLast", "VomitNbr"]] = 0

        if judg_calls["HA_umbrella"]:

            # cannot flatten if unionized
            assert prepr_calls["step15_HA"] != 1

            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("HA_verb", axis=1, inplace=True)
            # 92 is treated as 0
            try:
                df.loc[(df.HASeverity == 92) | (df.HAStart == 92),
                       ['HASeverity', 'HAStart']] = 0
            except AttributeError:
                df.loc[df.HASeverity == 92, 'HASeverity'] = 0

        if judg_calls["Seiz_umbrella"]:

            # cannot flatten if unionized
            assert prepr_calls["step16_Seiz"] != 1

            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("Seiz", axis=1, inplace=True)
            # 92 is treated as 0
            try:
                df.loc[(df.SeizLen == 92) | (df.SeizOccur == 92),
                       ['SeizLen', 'SeizOccur']] = 0
            except AttributeError:
                df.loc[df.SeizLen == 92, 'SeizLen'] = 0

        if judg_calls["LOC_umbrella"]:
            # cannot disambiguate if unclear had been assigned to 1
            assert not prepr_calls["step17_cautiousUncl"]

            # "flatten" - drop the umbrella variable, all children are equal
            df.drop("LOCSeparate", axis=1, inplace=True)
            # assign new category to 92 - no LOC
            df.loc[df.LocLen == 92, 'LocLen'] = 0

        if judg_calls["GCS"]:
            # include all three GCS scores, minus GCSTotal, and recode as 0/1
            df.drop("GCSTotal", axis=1, inplace=True)
            df['GCSVerbal'].replace((5, 4), (1, 0), inplace=True)
            df['GCSMotor'].replace((6, 5), (1, 0), inplace=True)
            df['GCSEye'].replace((4, 3), (1, 0), inplace=True)

        if not judg_calls["GCS"]:
            # include only the GCS total score, and recode as 0/1
            df.drop(["GCSVerbal", "GCSEye", "GCSMotor"], axis=1, inplace=True)
            df['GCSTotal'].replace((15, 14), (1, 0), inplace=True)

        # binarize categoricals
        #  set correct type on categoricals
        for col in df:
            if (col not in (self.get_meta_keys() + ["AgeinYears"])) \
                    & (len(df[col].unique()) > 2):
                # so the names aren't feature_92.0
                df[col] = df[col].astype(int)
                df[col] = df[col].astype('category')

        # there shouldn't be N/A outside of meta columns
        cols = [c for c in df.columns if
                isinstance(df.loc[:, c].dtype, pd.CategoricalDtype) and
                (c not in (self.get_meta_keys() + ["AgeinYears"]))]
        df = pd.get_dummies(df, dummy_na=False, columns=cols)

        # remove any col that has constant value in all observations
        if judg_calls["remove_constVal"]:
            df.drop(df.columns[df.nunique() <= 1], axis=1, inplace=True)

        assert np.sum(np.sum(pd.isna(df.drop(self.get_meta_keys(), axis=1)))) == 0, \
            "N/As present after extracting features!"

        return df

    def get_outcome_name(self) -> str:
        return 'outcome'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'tbi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        return ["Gender", "Race"]  # keys which are useful but not used for prediction

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
                    # Unionization policies:
                    # 1: unionize - impute parent from children, drop children
                    # 2: mixed: keep (some) children & parent, impute parent from the children kept
                    # 3: no: no imputation, keep all children, drop those with N/A in any child

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
                    "step19_Drugs"       : [False, True],
                    "step20_ActNormal"   : [True, False],

                },
                'extract_features': {
                    "HEMA_umbrella"   : [False, True],
                    "SFxPalp_umbrella": [False, True],
                    "SFxBas_umbrella" : [False, True],
                    "AMS_umbrella"    : [False, True],
                    "Clav_umbrella"   : [False, True],
                    "NeuroD_umbrella" : [False, True],
                    "Vomit_umbrella"  : [False, True],
                    "HA_umbrella"     : [False, True],
                    "Seiz_umbrella"   : [False, True],
                    "LOC_umbrella"    : [False, True],
                    "GCS"             : [True, False],
                    "remove_constVal" : [True, False]
                },
            }

        return judg_calls

    def get_judgement_calls_dictionary_default(self) -> Dict:
        """
         Returns the whole judgement calls dictionary with default values

        Returns
        -------
        Dict[str, list]
            Judgement calls dictionary.

        """

        # return the default
        return {fname: {k: v[0] for k, v in d.items()} for fname, d
                in self.get_judgement_calls_dictionary().items()}

    # NOTE: for quick reference - this is what's inherited and gets run:
    # NOTE: can actually override it if extra judgement call functionality needed!

    def get_judgement_calls_current(self) -> Dict[str, list]:
        """
         Returns the sub-dictionary of judgement calls for the calling function
         with default values, using inspection.

        Returns
        -------
        Dict[str, list]
            Judgement calls dictionary.

        """

        calling_func = inspect.currentframe().f_back.f_code.co_name
        judg_calls = self.get_judgement_calls_dictionary()[calling_func]

        # return the default
        return {k: v[0] for k, v in judg_calls.items()}

    def get_data(self, save_csvs: bool = False,
                 data_path: str = rulevetting.DATA_PATH,
                 load_csvs: bool = False,
                 run_perturbations: bool = False, **kwargs) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """Runs all the processing and returns the data.
        This method does not need to be overriden.

        Params
        ------
        save_csvs: bool, optional
            Whether to save csv files of the processed data
        data_path: str, optional
            Path to all data
        load_csvs: bool, optional
            Whether to skip all processing and load data directly from csvs
        run_perturbations: bool, optional
            Whether to run / save data pipeline for all combinations of judgement calls

        Returns
        -------
        df_train
        df_tune
        df_test
        """

        PROCESSED_PATH = oj(data_path, self.get_dataset_id(), 'processed')
        if load_csvs:
            return tuple([pd.read_csv(oj(PROCESSED_PATH, s), index_col=0)
                          for s in ['train.csv', 'tune.csv', 'test.csv']])
        np.random.seed(0)
        random.seed(0)
        CACHE_PATH = oj(data_path, 'joblib_cache')
        cache = Memory(CACHE_PATH, verbose=0).cache
        kwargs = self.get_judgement_calls_dictionary()
        default_kwargs = {}
        for key in kwargs.keys():
            func_kwargs = kwargs[key]
            default_kwargs[key] = {k: func_kwargs[k][0]  # first arg in each list is default
                                   for k in func_kwargs.keys()}

        print('kwargs', default_kwargs)
        if not run_perturbations:
            cleaned_data = cache(self.clean_data)(data_path=data_path, **default_kwargs)
            preprocessed_data = cache(self.preprocess_data)(cleaned_data, **default_kwargs)
            extracted_features = cache(self.extract_features)(preprocessed_data, **default_kwargs)
            df_train, df_tune, df_test = cache(self.split_data)(extracted_features)
        # TODO: handle exceptions
        elif run_perturbations:
            data_path_arg = init_args([data_path], names=['data_path'])[0]
            clean_set = build_Vset('clean_data', self.clean_data, param_dict=kwargs, cache_dir=CACHE_PATH)
            cleaned_data = clean_set(data_path_arg)
            preprocess_set = build_Vset('preprocess_data', self.preprocess_data, param_dict=kwargs,
                                        cache_dir=CACHE_PATH)
            preprocessed_data = preprocess_set(cleaned_data)
            extract_set = build_Vset('extract_features', self.extract_features, param_dict=kwargs,
                                     cache_dir=CACHE_PATH)
            extracted_features = extract_set(preprocessed_data)
            split_data = Vset('split_data', modules=[self.split_data])
            dfs = split_data(extracted_features)
        if save_csvs:
            os.makedirs(PROCESSED_PATH, exist_ok=True)

            if not run_perturbations:
                for df, fname in zip([df_train, df_tune, df_test],
                                     ['train.csv', 'tune.csv', 'test.csv']):
                    meta_keys = rulevetting.api.util.get_feat_names_from_base_feats(df.keys(), self.get_meta_keys())
                    df.loc[:, meta_keys].to_csv(oj(PROCESSED_PATH, f'meta_{fname}'))
                    df.drop(columns=meta_keys).to_csv(oj(PROCESSED_PATH, fname))
            if run_perturbations:
                for k in dfs.keys():
                    if isinstance(k, tuple):
                        os.makedirs(oj(PROCESSED_PATH, 'perturbed_data'), exist_ok=True)
                        perturbation_name = str(k).replace(', ', '_').replace('(', '').replace(')', '')
                        perturbed_path = oj(PROCESSED_PATH, 'perturbed_data', perturbation_name)
                        os.makedirs(perturbed_path, exist_ok=True)
                        for i, fname in enumerate(['train.csv', 'tune.csv', 'test.csv']):
                            df = dfs[k][i]
                            meta_keys = rulevetting.api.util.get_feat_names_from_base_feats(df.keys(),
                                                                                            self.get_meta_keys())
                            df.loc[:, meta_keys].to_csv(oj(perturbed_path, f'meta_{fname}'))
                            df.drop(columns=meta_keys).to_csv(oj(perturbed_path, fname))
                return dfs[list(dfs.keys())[0]]

        return df_train, df_tune, df_test


if __name__ == '__main__':
    # NOTE: for development
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

    judg_calls = self.get_judgement_calls_dictionary_default()
    judg_calls["preprocess_data"]["step19_Drugs"] = True
    judg_calls["preprocess_data"]["step20_ActNormal"] = False

    prep_data = self.preprocess_data(cleaned_data, **judg_calls)
    final_data = self.extract_features(prep_data, **judg_calls)

    # dset = Dataset()
    # df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=False)

    # print('successfuly processed data\nshapes:',
    #       df_train.shape, df_tune.shape, df_test.shape,
    #       '\nfeatures:', list(df_train.columns))
