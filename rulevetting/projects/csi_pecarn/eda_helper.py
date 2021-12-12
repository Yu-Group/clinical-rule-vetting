from os.path import join as oj

import numpy as np
import pandas as pd
import re

'''
This contains altered versions of the functions from `helper.py` which are used to generate Will's EDA plots
Currently this is only an imporved version of extract_numeric_data which ignores categorical covar. automatically
'''

def extract_numeric_data(input_df):
    '''
    This function returns a dataframe with all character columns dropped.
    Character variables which can be converted to binary such as 'Y'/'N' are mutated and kept
    Column names in categorical_covariates are automatically detected unchachanged by this method
    '''
     
    input_df = input_df.apply(pd.to_numeric, errors='ignore')
    
    numeric_data = input_df.select_dtypes([np.number]) # separate data that is already numeric
    char_data = input_df.select_dtypes([np.object]) # gets columns encoded as strings
    
    binary_data = pd.DataFrame(index=input_df.index) # init with study subject ID as index
    
    # add the suffix `_binary` to numeric variables with only 0 and 1 as non-nan inputs
    
    binary_numeric_cols = [col_name for col_name in numeric_data.columns\
                           if np.isin(numeric_data[col_name].dropna().unique(), [0, 1]).all()]
    
    numeric_data.columns = [col_name + '_binary' if col_name in binary_numeric_cols else col_name\
                                for col_name in numeric_data.columns]
    
    # convert text inputs to binary
    for column in char_data:
        if 'txt' in column: continue # don't process long-form text columns
        char_column = char_data[column] # select column
        unique_values = pd.unique(char_column) # get unique entries
        
        # encode yes as 1, no as 0
        if (('Y' in unique_values)|('A' in unique_values)|('YND' in unique_values)) & \
        (('N' in unique_values)|('ND' in unique_values) & ('Race' not in column)):
            conditions  = [char_column == 'Y',char_column == 'YD',char_column == 'YND',char_column == 'A',char_column == 'N']
            encodings = [1,1,1,1,0]
            binary_encoded = np.select(conditions, encodings, default=np.nan)
            col_name = column+"_binary"
            binary_data.loc[:,col_name] = binary_encoded.copy()
                
        # for clarity we convert the strings of post hoc outcomes into binary in the following loop
        elif (('INTUB' in unique_values)&('NOTUB' in unique_values)):
            conditions  = [char_column == 'Y', char_column == 'INTUB',char_column == 'EXTUB',char_column == 'NOTUB']
            encodings = [1,1,1,0]
            binary_encoded = np.select(conditions, encodings, default=np.nan)
            col_name = column+"_binary"
            binary_data.loc[:,col_name] = binary_encoded.copy()
        
        else: binary_data.loc[:,column] = char_column.copy()
    
    # add in newly created binary columns and removed categorical ones 
    output_df = pd.merge(numeric_data,binary_data,how="left",left_index=True,right_index=True)

    return output_df