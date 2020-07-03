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

def iwfm_date_to_datetime(iwfm_date_string):
    ''' converts an IWFM date of format MM/DD/YYYY_HH:MM to datetime

    IWFM dates are good until the date provided. In many cases, the date provided is 
    the last day of the month and time is 24:00 hours which is not valid in python.
    
    
    for example, 09/30/2019_24:00 would be the same as 10/01/2019 00:00 in python, 
    but raises a ValueError.

    This function handles the exception if the time is 24:00 and the time is removed. 
    
    Parameters
    ----------
    iwfm_date_string : str
        date with format MM/DD/YYYY_HH:MM

    Returns
    -------
    datetime
        date in datetime format
    '''
    # attempt to convert the raw IWFM date string
    try:
        dt = datetime.datetime.strptime(iwfm_date_string, '%m/%d/%Y_%H:%M')
    except ValueError:
        try:
            date_string = iwfm_date_string.split('_')[0]
            dt = datetime.datetime.strptime(date_string, '%m/%d/%Y')
        except ValueError:
            raise
        else:
            return dt
    else:
        return dt

def last_day_of_month(any_day):
    ''' returns the date for the last day in the month for any date

    Parameters
    ----------
    any_day : datetime
        any valid datetime

    Returns
    -------
    datetime
        date of the last day of the month
    '''
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    
    return next_month - datetime.timedelta(days=next_month.day)