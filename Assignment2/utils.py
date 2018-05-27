from operator import itemgetter
from sklearn.utils import resample
import pandas as pd
import csv

def write_submission(recommendations, submission_file):
    """
    Function which writes submission, ordered on the probability obtained by the model.
    The columns are SearchId, PropertyId and Relevance    
    """
    global rows
    submission_path = submission_file
    rows = [(srch_id, prop_id, relevance)
        for srch_id, prop_id, relevance, rank_float
        in sorted(recommendations, key=itemgetter(0,3))]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    writer.writerow(("SearchId", "PropertyId", "Relevance"))
    writer.writerows(rows)



def balance_dataset(train, downsampling_rate):
    
    # Separate majority and minority classes
    df_majority = train[train.relevance==0]
    df_minority = train[train.relevance==5]
    df_minority2 = train[train.relevance==1]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,    # sample without replacement
                                     n_samples=df_minority.shape[0]*downsampling_rate,     # to match minority class
                                     random_state=123) # reproducible results


    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority, df_minority2])

    # Display new class counts
    df_downsampled.relevance.value_counts()

    df_downsampled['relevance'].hist()
    #df_downsampled['relevance'].value_counts()
    train = df_downsampled
    train = train.sort_values('srch_id')
    return train



def define_features():
    """
    removes features that should not be used for training
    eg. duplicates/raw features/position


    :return: selected features: list with features to choose
    """

  

    selected_features = ['random_bool',
     'value_for_money_star_norm_srch_id',
     'visitor_hist_adr_usd',
     'value_for_money_norm_srch_id',
     'price_usd_norm_srch_id',
     'srch_query_affinity_score',
     'prop_location_score2_norm_srch_id',
     'comp1_rate_percent_diff_log',
     'prop_location_score_mean',
     'prop_location_score1_norm_srch_id',
     'srch_room_count',
     'prop_review_score_norm_srch_id',
     'price_diff_from_historic_mean',
     'visitor_hist_starrating',
     'comp7_rate_percent_diff_signed',
     'comp6_rate_percent_diff_signed',
     'comp1_rate_percent_diff_signed',
     'srch_adults_count',
     'prop_location_score2_norm',
     'price_rank',
     'star_rank',
     'prop_location_score2',
     'prop_review_score_norm_srch_destination_id',
     'prop_log_historical_price_norm_srch_id',
     'prop_brand_bool',
     'prop_starrating_norm_srch_destination_id',
     'prop_country_id',
     'visitor_location_country_id',
     'value_for_money_star_norm_srch_destination_id',
     'prop_starrating',
     'prop_review_score',
     'srch_children_count_norm_srch_id',
     'srch_length_of_stay',
     'price_usd_norm_prop_id',
     'site_id',
     'value_for_money_norm_srch_destination_id',
     'srch_saturday_night_bool',
     'price_usd_norm_srch_destination_id',
     'prop_starrating_monot',
     'value_for_money_star',
     'srch_destination_id',
     'value_for_money',
     'srch_query_affinity_score_norm_srch_id',
     'price_usd_log',
     'month',
     'price_usd_norm_srch_saturday_night_bool',
     'srch_booking_window',
     'price_usd',
     'promotion_flag',
     'srch_adults_count_norm_srch_id',
     'orig_destination_distance',
     'prop_location_score1',
     'srch_children_count',
     'prop_log_historical_price',
     'star_diff_from_historic_mean',
     'prop_location_score1_norm',
     'comp5_rate_percent_diff_signed',
     'comp2_inv',
     'comp6_inv',
     'comp_rate_percent_diff_mean',
     'comp7_inv',
     'comp3_rate_percent_diff_signed',
     'comp8_rate_percent_diff_signed',
     'comp5_inv', 
     'prop_id']


    TO_DROP=[
     'comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate',
     'comp6_rate','comp7_rate','comp8_rate',
     'comp6_rate_percent_diff',
     'comp4_rate_percent_diff',
     'comp7_rate_percent_diff',
     'comp1_rate_percent_diff',
     'comp3_rate_percent_diff',
     'comp2_rate_percent_diff',
     'comp8_rate_percent_diff',
     'comp5_rate_percent_diff',
     'comp1_rate_percent_diff_signed_norm',
     'comp2_rate_percent_diff_signed_norm',
     'comp4_rate_percent_diff_signed_norm',
     'comp8_rate_percent_diff_signed_norm',
     'comp3_rate_percent_diff_signed_norm',
     'comp5_rate_percent_diff_signed_norm',
     'comp6_rate_percent_diff_signed_norm',
     'comp7_rate_percent_diff_signed_norm',
     'prop_location_score1',
     'prop_location_score2',
     'srch_id',
     'position',
     'log_price_usd']

    selected_features = list(set(selected_features))
    TO_DROP = list(set(TO_DROP))

    for feat in list(TO_DROP):
        if feat in selected_features:
            selected_features.remove(feat)
    return selected_features
