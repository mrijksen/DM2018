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

