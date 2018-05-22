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
    all_cols = df.columns.drop(['booking_bool', 'click_bool', 'gross_bookings_usd', 'position'])


    for feature in all_cols:
        df[feature][df[feature]>df[feature].quantile(0.99)] = df[feature].quantile(0.99)


    for feature in all_cols:
        df[feature][df[feature]<df[feature].quantile(0.01)] = df[feature].quantile(0.01)
    return(df)

def convert_type(df):

    return df


def create_composite_features(df):
    #df['date_time']= pd.to_datetime(df['date_time'])
    #data.date_time.map(lambda x: x.month)

    #df['season'] = df.date_time.apply(lambda dt: (dt.month%12 + 3)//3)


    # Rank within the same srch id
    #
    df['price_rank'] = df.groupby(['srch_id'])['price_usd'].rank(method='dense')
    df['star_rank'] = df.groupby(['srch_id'])['price_usd'].rank(method='dense')

    df['value_for_money']=df.price_usd/df.prop_review_score
    df['value_for_money'] = df.prop_review_score/np.log10(df.price_usd)

    # Merge location score
    # Found to perform better after filling in the nulls..
    feature = "prop_location_score1"
    df[feature][df[feature].isnull()] = df[feature].median()
    df[feature+"_norm"] = (df[feature] - df[feature].mean()) / (df[feature].std())
    feature = "prop_location_score2"
    df[feature][df[feature].isnull()] = df[feature].median()
    df[feature+"_norm"] = (df[feature] - df[feature].mean()) / (df[feature].std())
    df["prop_location_score_mean"] =  df[['prop_location_score2_norm', 'prop_location_score1_norm']].mean(axis=1)


    # Logs
    feature = "comp1_rate_percent_diff"
    df["comp1_rate_percent_diff_log"]=np.log10(df["comp1_rate_percent_diff"])

    return df

def normalize(df,feature, group):
    # important: remove outliers before the normalization
    # otherwise stdev and mean will be skewed

    normalized = df[[feature,group]].groupby(group).transform(lambda x: (x - x.mean()) / (x.std()))

    return normalized

def normalize_within_group(df):

    group = 'srch_id'
    # Normalize
    feature = "price_usd"
    df['price_usd_norm_srch_id'] = normalize(df, feature, group)

    df['log_price_usd'] = np.log10(df.price_usd)
    feature = 'log_price_usd'

    feature = "value_for_money"
    df['value_for_money_norm_srch_id'] = normalize(df, feature, group)
    feature = "prop_starrating"
    group = 'srch_destination_id'
    #df['prop_starrating_norm_srch_id'] = normalize(df, feature, group)

    group = 'prop_id'
    df['price_usd_norm_prop_id'] = normalize(df, feature, group)

    return df


def missing_values(df):
    # continuous

    # discrete

    return df