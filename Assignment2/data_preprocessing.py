__author__ = 'efiathieniti'

# Includes functions for data cleaning operations

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os



def remove_outliers(df):
    # removes outliers
    # if normal, use standard deviation
    # if not normal uses percentiles

    return(df)


def convert_type(df):

    return df


def create_composite_features(df):
    df['date_time']= pd.to_datetime(df['date_time'])
    #data.date_time.map(lambda x: x.month)

    df['season'] = df.date_time.apply(lambda dt: (dt.month%12 + 3)//3)


    # Rank within the same srch id
    #
    df['price_rank'] = df.groupby(['srch_id'])['price_usd'].rank(method='dense').astype(int)
    df['star_rank'] = df.groupby(['srch_id'])['price_usd'].rank(method='dense').astype(int)

    df['value_for_money']=df.price_usd/df.prop_review_score
    df['value_for_money'] = df.prop_review_score/df.price_usd

    return df


def normalize_within_group(df):

    # Normalize
    df['price_usd_normalized'] = df[['price_usd','srch_id']].groupby('srch_id').transform(lambda x: (x - x.min()) / (x.max()-x.min()))
    return df


def missing_values(df):
    # continuous

    # discrete

    return df