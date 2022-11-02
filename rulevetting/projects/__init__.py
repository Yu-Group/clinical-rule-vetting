import logging

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)


def one_hot_encode_df(df, numeric_cols):
    """Transforms categorical features in dataframe 
    Returns 
    -------
    one_hot_df: pd.DataFrame - categorical vars are one-hot encoded 
    """
    # grab categorical cols with >2 unique features
    categorical_cols = [
        col for col in df.columns if not set(df[col].unique()).issubset({0.0, 1.0, 0, 1}) and col not in numeric_cols]
    one_hot_df = pd.get_dummies(df.astype(str), columns=categorical_cols)
    
    return one_hot_df
