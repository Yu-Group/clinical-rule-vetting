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


class Dataset(DatasetTemplate):
    def clean_data(self, data_path: str = rulevetting.DATA_PATH, **kwargs) -> pd.DataFrame:
        raw_data_path = oj(data_path, self.get_dataset_id(), 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        # all the fnames to be loaded and searched over        
        fnames = sorted([fname for fname in os.listdir(raw_data_path) if 'csv' in fname])
        # read through each fname and save into the r dictionary
        r = {}
        print('read all the csvs...\n', fnames)
        if len(fnames) == 0:
            print('no csvs found in path', raw_data_path)
        
        # replace studysubjectid cases with id
        for fname in tqdm(fnames):
            df = pd.read_csv(oj(raw_data_path, fname), encoding="ISO-8859-1")
            df.columns = [re.sub('StudySubjectID','id',x,flags=re.IGNORECASE) for x in df.columns]
            df.columns = [re.sub('SITE','site',x) for x in df.columns]
            df.columns = [re.sub('CaseID','case_id',x,flags=re.IGNORECASE) for x in df.columns]
            df.columns = [re.sub('ControlType','control_type',x,flags=re.IGNORECASE) for x in df.columns]
            df.columns = [re.sub('CSpine','CervicalSpine',x) for x in df.columns]
            df.columns = [re.sub('subinj_','SubInj_',x) for x in df.columns]
                            
            assert ('id' in df.keys())
            df = df.set_index(['id','case_id','site','control_type']) # use a multiIndex
            r[fname] = df

        # Get filenames we consider in our covariate analysis
        # We do not consider radiology review data at this time

        df_features = r['analysisvariables.csv'] # keep `site`, `case id`, and `control type` covar from first df
        
        
        # New Data Source
        # the second most useful predictive covariates are those collected at the study site
        ss_data = r['clinicalpresentationsite.csv']
        
        # first we hand-select features after conversations with Dr. Devlin about how a patient arrives to the ED
        # note we do not include `DxCspineInjury` at Dr. Devlin's suggestion despite its strong predictive power
        ss_arrival_features = ['ModeArrival','ReceivedInTransfer','PtAmbulatoryPriorArrival','CervicalSpineImmobilization',\
                           'ArrPtIntub']
        ss_arrival_data = ss_data[ss_arrival_features]
        df_features = pd.merge(df_features,ss_arrival_data,how="left",left_index=True,right_index=True)
        
        # next get AVPU and GCS test results
        ss_eval_results = ['GCSEye','MotorGCS','VerbalGCS','TotalGCS','AVPUDetails']
        ss_eval_data = ss_data[ss_eval_results]
        df_features = pd.merge(df_features,ss_eval_data,how="left",left_index=True,right_index=True)
        
        # the analysis variables consider neck pain and we case a wider net because young children are not good
        # at localizing where pain is coming from. Tenderness is pain observed by doctor, not self-reported
        ss_pain_features = ['PtCompPainNeck','PtCompPainFace','PtCompPainHead','PtCompPainNeckMove',\
                            'PtTenderNeck','PtTenderFace','PtTenderHead']
        ss_pain_data = ss_data[ss_pain_features]
        df_features = pd.merge(df_features,ss_pain_data,how="left",left_index=True,right_index=True)
                
        # other covariates about localized neck tenderness, major injuries, and focal neurological findings are
        # summarzied by analysis variables already
        
        # features about the study site outcomes are not used in prediciton but are useful to test the efficacy of our preds.
        ss_outcome_precautions = [col for col in ss_data if col.startswith('CervicalSpinePrecautions')]
        ss_outcome_intervention = [col for col in ss_data if col.startswith('IntervForCervical')]
        ss_outcomes = [col for col in ss_data if col.startswith('OutcomeStudySite')]
        ss_outcomes_other = ['MedsGiven','IntubatedSS','LongTermRehab','TrfToLongTermRehab']
        posthoc_outcomes_all = ss_outcome_precautions+ss_outcome_intervention+ss_outcomes+ss_outcomes_other
        posthoc_features  = ss_data[posthoc_outcomes_all]
        posthoc_features.columns = 'posthoc_' + posthoc_features.columns.astype(str)
        df_features = pd.merge(df_features,posthoc_features,how="left",left_index=True,right_index=True)
        
        # New Data Source
        # as the outside and field datasets contain the same covariates collected before arrival at the study site
        # we only use this data as a robustness check. This decision was approved by Dr. Devlin
        # we do not include medsgiven and a prior hospital or EMS, nor do we include the GCS or AVPU score outside the study site
        # TODO: justify
            
        all_auxiliary_features = ss_arrival_features + ss_pain_features
        outside_data = r['clinicalpresentationoutside.csv']
        ems_data = r['clinicalpresentationfield.csv']
       
        outside_covariates = [col for col in outside_data if col in all_auxiliary_features]
        ems_covariates = [col for col in ems_data if col in all_auxiliary_features]
        # TODO: sedatives and GCS?
        
        outside_included_data = outside_data[outside_covariates]
        outside_included_data.columns = outside_included_data.columns.astype(str) + '_outside'
        ems_included_data = ems_data[ems_covariates]
        ems_included_data.columns = ems_included_data.columns.astype(str)  + '_ems'
        df_features = pd.merge(df_features,outside_included_data,how="left",left_index=True,right_index=True)
        df_features = pd.merge(df_features,ems_included_data,how="left",left_index=True,right_index=True)
        
        
        # New Data Source
        # the only demographic features we consider are AgeInYear and Gender
        # we include Race and PayorType as meaningful posthoc_covariates to evaluate fairness
        demographic_data = r['demographics.csv']
        demographic_features = ['AgeInYears','Gender','Race','PayorType']
        demographic_included_data = demographic_data[demographic_features].rename\
        (columns={"Race": "posthoc_Race", "PayorType": "posthoc_PayorType"})
        df_features = pd.merge(df_features,demographic_included_data,how="left",left_index=True,right_index=True)
        
        
        # New Data Source
        # The injry classification contains posthoc data to evaluate how our predicitons vary by injury type
        injuryclassification_data = r['injuryclassification.csv']
        ic_summary_data = injuryclassification_data[["CSFractures","Ligamentoptions","CervicalSpineSignalChange"]]
        ic_summary_data.rename(columns={"CSFractures": "CervicalSpineFractures",\
                                        "Ligamentoptions": "LigamentInjury"},inplace=True)
        ic_summary_data.columns = 'posthoc_' + ic_summary_data.columns.astype(str)
        df_features = pd.merge(df_features,ic_summary_data,how="left",left_index=True,right_index=True)
        
        # New Data Source
        # The injry mechanism is well-summarized by analysis variables, which indidate high risk isntance.
        # We include some covariates from this dataset hich add nunace to the injury
        # such as wheter the patient was wearing a helmet
        injurymechanism_data = r['injurymechanism.csv']
        im_features = ['PassRestraint','Assault','ChildAbuse','helmet','FallDownStairs']
        im_included_data = injurymechanism_data[im_features]
        df_features = pd.merge(df_features,im_included_data,how="left",left_index=True,right_index=True)
    
    
        # New Data Source
        # Analysis variables summarizes a patient's medical history well, especially for predisposing conditions.
        # We include additional binary indicators which indicate any prior abnormal medical history
        medicalhistory_data = r['medicalhistory.csv']
        mh_features = ['HEENT','Cardiovascular','Respiratory','Gastrointestinal',\
                           'Musculoskeletal','Neurological','Medications']
        mh_included_data = medicalhistory_data[mh_features]
                
        df_features = pd.merge(df_features,mh_included_data,how="left",left_index=True,right_index=True)

        # New Data Source
        # Patients treated first at another hospital may have had radiology there. These covariates will help
        # determine if this information affects the radiology decisions made by study site doctors
        radiologyoutside_data = r['radiologyoutside.csv']
        radiologyoutside_included_data  = radiologyoutside_data[["Xrays","CTPerformed","MRIPerformed"]]
        radiologyoutside_included_data.columns = radiologyoutside_included_data.columns.astype(str) + '_outside'
        df_features = pd.merge(df_features,radiologyoutside_included_data,how="left",left_index=True,right_index=True)
        
        # New Data Source
        # For post-hoc evaluation we consider what radiological tests were ordered at the study site
        radiologysite_data= r['radiologysite.csv']
        radiologysite_icluded_data = radiologysite_data[["Xrays","CTPerformed","MRIPerformed"]]
        radiologysite_icluded_data.columns = 'posthoc_' + radiologysite_icluded_data.columns.astype(str) + '_site'
        df_features = pd.merge(df_features,radiologysite_icluded_data,how="left",left_index=True,right_index=True)

        
        # New Data Source
        # judgement call to use re-abstracted kappa infromation for appropriate units and features
        if kwargs['use_kappa']:
            kappa_data = r['kappa.csv']
            # drop kappa columns not in full dataset
            to_drop_kappa_cols = kappa_data.columns.difference(df_features.columns)
            kappa_data.drop(to_drop_kappa_cols, axis=1, inplace=True)
            # replace with kappa data at relavent locations
            df_features.loc[kappa_data.index,kappa_data.columns] = kappa_data
        
        
        print("{0} Raw Covariates Selected".format(df_features.shape[1]))
        return df_features

    def preprocess_data(self, cleaned_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        # list of categorical columns to ignore
        categorical_covariates = ['posthoc_Race','posthoc_PayorType',\
                                  'posthoc_OutcomeStudySite','posthoc_OutcomeStudySiteMobility','posthoc_OutcomeStudySiteNeuro']
        df = cleaned_data
        #
        # add a binary outcome variable for CSI injury 
        df.loc[:,'csi_injury'] = df.index.get_level_values('control_type').map(helper.assign_binary_outcome)

        # convert numeric columns encoded as strings
        numeric_as_str_cols = ['TotalGCS', 'ModeArrival']
        for col_name in numeric_as_str_cols: # .to_numeric only works on series
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce') # coerce makes non-numeric to NA
        
        df.loc[:,'EMSArrival'] = df.loc[:,'ModeArrival'].replace([1,2],[1,0]) # did patient arrive via EMS?
        df.drop(['ModeArrival'], axis=1, inplace=True)
        
        # correct encoding for indicator of cspine immobilization
        df.loc[:,'CervicalSpineImmobilization'] = df.loc[:,'CervicalSpineImmobilization'].replace([1,2,3],[1,1,0])
        
        # change binary variable label so that 1 is abnormal condition
        df.loc[:,'NonAmbulatory'] = df.loc[:,'ambulatory'].replace([1,0],[0,1])
        df.drop(['ambulatory'], axis=1, inplace=True)
        
        # change gender in to binary indicator for male (60% majority category)
        df.loc[:,'male'] = df.loc[:,'Gender'].replace(['M','F','ND'],[True,False,False])
        df.drop(['Gender'], axis=1, inplace=True)
        
        # drop uniformative columns which only contains a single value
        # should be 0
        no_information_columns = df.columns[df.nunique() <= 1]
        df.drop(no_information_columns, axis=1, inplace=True)
        assert(len(no_information_columns) == 0)
                 
        # create one-hot encoding of AVPU data
        avpu_columns = [col for col in df.columns if 'avpu' in col.lower()]
        df[avpu_columns] = df[avpu_columns].replace('N',np.NaN)

        df[avpu_columns] = 'AVPU_' + df[avpu_columns].astype(str)
        avpu_one_hot = pd.get_dummies(df[avpu_columns])
        df = df.drop(avpu_columns,axis = 1)
        df = df.join(avpu_one_hot)
        
        pd.options.mode.chained_assignment = None
        
        gcs_columns = [col for col in df.columns if 'gcs' in col.lower()]
        for gcs_col in gcs_columns:
            max_gcs = df[gcs_col].max()
            df[gcs_col][(df['AlteredMentalStatus'] == 0.0) & (pd.isna(df[gcs_col]))] = max_gcs
        
        #
        # Judgement call to fill ~2% of units without these outcomes as normal
        df['posthoc_OutcomeStudySiteMobility'][(pd.isna(df['posthoc_OutcomeStudySiteMobility']))] = 'N'
        df['posthoc_OutcomeStudySiteNeuro'][(pd.isna(df['posthoc_OutcomeStudySiteNeuro']))] = 'NR'
        
        pd.options.mode.chained_assignment = 'warn'
        df = helper.extract_numeric_data(df,categorical_covariates=categorical_covariates)
        
        df = helper.build_robust_binary_covariates(df)
        
        return df

    def extract_features(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # add engineered featuures
        df = preprocessed_data     
        df = helper.rename_values(df)
        df = helper.derived_feats(df)
        
        robust_columns = df.columns[df.columns.str.endswith('2')]\
            .drop(['posthoc_OutcomeStudySiteMobility2']) # endswith 2 regex is too strong
        nonrobust_columns = [col_name[:-1] for col_name in robust_columns] 
        # drop columns for jdugement call
        if kwargs['use_robust_av']: df.drop(nonrobust_columns, axis=1, inplace=True)
        else: df.drop(robust_columns, axis=1, inplace=True)
            
        '''
        # bin useful continuous variables age
        binning_dict = {}
        binning_dict['AgeInYears'] = (2,6,12)        
        df = helper.bin_continuous_data(df, binning_dict)
        ''' 

        return df
    
    def impute_data(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = preprocessed_data
        
        # impute missing binary variables with 0; this is justified because abnormal responses are encoded as 1
        # and we make a judgement call to assune that all relavent abnormal information is recorded
      
        
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        #print((df.isna().sum(axis = 0)/df.shape[0]).sort_values())
        #df = helper.impute_missing_binary(df, n=kwargs['frac_missing_allowed']) 
        
        # TODO: discuss with Gabriel
        # For now, impute subgroup GCS with maximum and recalcualte total
        
        df[['GCSEye','MotorGCS','VerbalGCS']] = \
            df[['GCSEye','MotorGCS','VerbalGCS']].apply(lambda col: col.fillna(col.max()), axis=0)
        df['TotalGCS'] = df['GCSEye'] + df['MotorGCS'] + df['VerbalGCS']
        
        df['GCS15'] = (df['TotalGCS']== 15).replace([True,False],[1,0])   
        
        '''
        df['GCS_NA_total'] = pd.isna(df['TotalGCS']).replace([True,False],[1,0])
        df['GCS_NA_eye'] = pd.isna(df['GCSEye']).replace([True,False],[1,0])
        df['GCS_NA_motor'] = pd.isna(df['MotorGCS']).replace([True,False],[1,0])
        df['GCS_NA_verbal'] = pd.isna(df['VerbalGCS']).replace([True,False],[1,0])
        
    
        for column in df.columns:
            char_column = df[column] # select column
            unique_values = pd.unique(char_column) # get unique entries
        '''
        
        df = helper.impute_missing_binary(df, n=kwargs['frac_missing_allowed']) 

        # df.fillna(0, inplace=True) # deprecated all NA filled by this step
        
        numeric_data = df.select_dtypes([np.number]) # separate data that is already numeric
        numeric_data = numeric_data.astype(float)
        char_data = df.select_dtypes([np.object]) # gets columns encoded as strings
        
        df = pd.merge(numeric_data,char_data,how="left",left_index=True,right_index=True)
        
        return df
    
    def split_data(self, preprocessed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        print('split_data kwargs', kwargs)
        
        col_names = ['id','case_id','site','control_type'] + list(preprocessed_data.columns.copy())
        df_train = pd.DataFrame(columns=col_names)
        df_train = df_train.set_index(['id','case_id','site','control_type'])
        df_tune = pd.DataFrame(columns=col_names)
        df_tune = df_tune.set_index(['id','case_id','site','control_type'])
        df_test = pd.DataFrame(columns=col_names)
        df_test = df_test.set_index(['id','case_id','site','control_type'])
        
        study_site_list = [i for i in range(1,18)]
        print(kwargs['control_types'])
        selected_control_types = ['case']+kwargs['control_types']
        
        for ss in study_site_list:
            for ct in selected_control_types:
                split_subset = preprocessed_data.xs((ss, ct), level=('site','control_type'), drop_level=False) # subset to split
                
                # do the splitting below
                split_data = np.split(split_subset.sample(frac=1, random_state=42),
                                      [int(.6 * len(split_subset)), int(.8 * len(split_subset))])
                df_train = pd.concat([df_train,split_data[0]])
                df_tune = pd.concat([df_tune,split_data[1]])
                df_test = pd.concat([df_test,split_data[2]])
                
        return tuple([df_train,df_tune,df_test])

    def get_outcome_name(self) -> str:
        return 'csi_injury'  # return the name of the outcome we are predicting

    def get_dataset_id(self) -> str:
        return 'csi_pecarn'  # return the name of the dataset id

    def get_meta_keys(self) -> list:
        return ['Race', 'InitHeartRate', 'InitSysBPRange']  # keys which are useful but not used for prediction

    def get_judgement_calls_dictionary(self) -> Dict[str, Dict[str, list]]:
        return {
            'clean_data': { 
                'use_kappa':[False, True],
            },
            'preprocess_data': {             
            },'extract_features': { 
                # some variables from `AnaylsisVariables.csv` end with a 2
                # using positive findings from field or outside hospital documentation these have 
                # the response to YES from NO or MISSING. The Leonard (2011) study considers them more robust
                # use mirror this perturbation for our own derived features
                'use_robust_av':[False, True],
            },
            'impute_data': { 
                # drop units with missing this percent of analysis variables or more

                'frac_missing_allowed': [0.05, 0.1],
            },
            'split_data': {
                # drop cols with vals missing this percent of the time
                'control_types': [['ran','moi','ems']],
            }
        }
    
    def get_data(self, save_csvs: bool = False,
                 data_path: str = rulevetting.DATA_PATH,
                 load_csvs: bool = False,
                 run_perturbations: bool = False,
                 control_types=['ran','moi','ems']) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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
        control_types: list of str, optional
            Which control types (Random, Mechanism of Injury, EMS) to include
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

        if not run_perturbations:
            pass
            cleaned_data = cache(self.clean_data)(data_path=data_path, **default_kwargs['clean_data'])
            preprocessed_data = cache(self.preprocess_data)(cleaned_data, **default_kwargs['preprocess_data'])
            featurized_data = cache(self.extract_features)(preprocessed_data, **default_kwargs['extract_features'])
            imputed_data = cache(self.impute_data)(featurized_data, **default_kwargs['impute_data'])
            df_train, df_tune, df_test = cache(self.split_data)(imputed_data, **{'control_types': control_types})
        elif run_perturbations:
            data_path_arg = init_args([data_path], names=['data_path'])[0]
            clean_set = build_Vset('clean_data', self.clean_data, param_dict=kwargs['clean_data'], cache_dir=CACHE_PATH)
            cleaned_data = clean_set(data_path_arg)
            preprocess_set = build_Vset('preprocess_data', self.preprocess_data, param_dict=kwargs['preprocess_data'],
                                        cache_dir=CACHE_PATH)
            preprocessed_data = preprocess_set(cleaned_data)
            extract_set = build_Vset('extract_features', self.extract_features, param_dict==kwargs['split_data'],
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
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=True)
    print('successfuly processed data\nshapes:',
          df_train.shape, df_tune.shape, df_test.shape,
          '\nfeatures:', list(df_train.columns))
