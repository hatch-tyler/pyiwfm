import os
import datetime
import numpy as np
import pandas as pd

def dataframe_to_structured_array(df):
    ''' converts a pandas dataframe object to a structured numpy array
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame Object to convert

    Returns
    -------
    structured ndarray 
    '''
    columns = np.array(df.columns)
    dtypes = [(name, df[name].dtype) for name in columns]
    
    return np.array(list(df.itertuples(index=False)), dtype=dtypes)