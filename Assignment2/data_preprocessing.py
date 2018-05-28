__author__ = 'efiathieniti'

# Includes functions for data cleaning operations

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os



def remove_outliers(df, all_cols):
    # removes outliers
    # if normal, use standard deviation
    # if not normal uses percentiles


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

    df['price_usd_log'] = np.log10(df.price_usd)

    # Rank within the same srch id
    #
    df['price_rank'] = df.groupby(['srch_id'])['price_usd'].rank(method='dense')
    df['star_rank'] = df.groupby(['srch_id'])['price_usd'].rank(method='dense')


    df['value_for_money_star']=df.prop_starrating/np.log10(df.price_usd)
    df['value_for_money'] = df.prop_review_score/np.log10(df.price_usd)

    # Normalize
    df['prop_starrating_monot']=abs(df.prop_starrating - df.prop_starrating.mean())

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

    print("Created new features")

    comp_feats=['comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate']

    for feat in comp_feats:
        df[feat + "_percent_diff_signed"]=np.log10(df[feat+"_percent_diff"])*df[feat]

        feature =  feat + "_percent_diff_signed"

        df[feature+"_norm"]=(df[feature] - df[feature].mean()) / (df[feature].std())


    # Take an average of the comp1_rate_percent_diff after normalizing
    comp_feats_signed=[]
    for feat in comp_feats:
        comp_feats_signed.append(feat+'_percent_diff_signed')

    df['comp_rate_percent_diff_mean']=df[comp_feats_signed].mean(axis=1)


    df['price_diff_from_historic_mean']= df.price_usd - df.visitor_hist_adr_usd
    df['star_diff_from_historic_mean']= df.prop_starrating - df.visitor_hist_starrating


    return df

def normalize(df,feature, group):
    # important: remove outliers before the normalization
    # otherwise stdev and mean will be skewed

    normalized = df[[feature,group]].groupby(group).transform(lambda x: (x - x.mean()) / (x.std()))

    return normalized

def normalize_within_group(df):

    """
    Normalizes the hotel features with their mean within the srch_id
    :param df:
    :return: df
    """
    """
    :param df:
    :return:
    """

    feats_to_normalize = ['prop_location_score1',
               'prop_location_score2',
                'price_usd_log',
              # 'book_per_pcnt',
               'price_usd',
               'value_for_money',
               'value_for_money_star',
               'prop_review_score',
                'srch_adults_count',
                'srch_children_count',
               #'promo_per_procnt',
               'prop_log_historical_price',
               'srch_query_affinity_score']
               #'click_nobook_per_pcnt']

    group = 'srch_id'
    # Normalize
    for feature in feats_to_normalize:
        df[feature+"_norm_"+group] = normalize(df, feature, group)

    print("Normalized features with srch_id")
    group = "srch_destination_id"
    feats_to_normalize = ["price_usd",
                          "prop_starrating",
                          "prop_review_score",
                          "value_for_money",
                          "value_for_money_star"]

    for feature in feats_to_normalize:
        df[feature+"_norm_"+group] = normalize(df, feature, group)

    print("Normalized features with srch_destination_id")

    group = "srch_saturday_night_bool"
    feature = "price_usd"
    df[feature+"_norm_"+ group] = normalize(df, feature, group)

    group = 'prop_id'
    feature = 'price_usd'
    df['price_usd_norm_prop_id'] = normalize(df, feature, group)

    print("Normalized features with srch_saturday_night_bool and prop_id")

    return df


def missing_values(df, all_cols):
    # continuous
    """

    :param df:
    :param all_cols: columns to fill in values
    :return:
    """


    for feature in all_cols:
        df[feature][df[feature].isnull()] = df[feature].median()


    return df
