
# coding: utf-8

# In[32]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
#Models


import data_preprocessing



# In[104]:

import imp
data_preprocessing = imp.reload(data_preprocessing)
import time


# In[86]:

output_path = "./Plots/"


overall_start_time = time.time()

training_data = True
select_n_rows = True
n_select =None

if training_data:
    data_original = pd.read_csv('./Data Mining VU data/training_set_VU_DM_2014.csv')
else:
    data_original = pd.read_csv('./Data Mining VU data/test_set_VU_DM_2014.csv')

# In[88]:

len(data_original)


# In[89]:
if select_n_rows:
    n_select = 4000000
    df = data_original.iloc[n_select:]
    df_original = data_original.iloc[n_select:]
else:
    df = data_original
    df_original = data_original



# In[ ]:




# #  Feature Engineering

# ## Composite features
# Rank by group
# 1. Extract season: autumn, summer, winter 
# 

# In[90]:

# Create the relevance class
if training_data:
    df["relevance"] = df["booking_bool"]+ df["click_bool"]
    df['relevance'] = df['relevance'].map({0:0, 1:1, 2:5})
    class_to_plot = "relevance"



df['date_time']= pd.to_datetime(df['date_time'])
df["month"] = df["date_time"].dt.month
df = df.drop(['date_time'], axis=1)


if training_data:
    all_cols=df.columns.drop(['booking_bool', 'click_bool', 'gross_bookings_usd', 'position'])
else:
    all_cols=df.columns

df = data_preprocessing.missing_values(df, all_cols)


# 

# In[94]:

if training_data:
    all_cols=df.columns.drop(['booking_bool', 'click_bool', 'gross_bookings_usd', 'position'])
else:
    all_cols=df.columns

# ### NEW FEATURES

# In[ ]:




# ### Outliers
# For features with high value outliers cap to maximum

# In[95]:

# If testing feature engineering use balance_dataset, otherwise use the whole 
# dataset
#df = utils.balance_dataset(df, 3)
df = data_preprocessing.remove_outliers(df, all_cols)

create_plots=None


# In[96]:


if create_plots:
    for feature in all_cols:
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)  
        df_before[[feature]].hist(bins=10, ax=ax2)
        df[[feature]].hist(bins=10, ax=ax1)
        plt.savefig(output_path + "hist_remove_outliers_%s.png"%feature, format='png')

        plt.show()



# In[97]:


start = time.time()


# In[105]:

# about 15 mins for the full dataset.. 

df = data_preprocessing.create_composite_features(df)
# Remove outliers before normalizing?

df = data_preprocessing.normalize_within_group(df)

print(time.time()-start)


# ### Set Missing values to median
# Do this after new features are created?

# In[106]:

#df = data_preprocessing.remove_outliers(df, all_cols)



# ### Calculate correlation and mutual information/information gain

# ### Histograms

# ## Distribution of each feature for booked and not booked hotels
# Helps find the most discriminative features

# In[21]:

create_plots = None


# In[22]:

if create_plots:
    for feature in all_cols:
        try:
            fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))  
            df.groupby(class_to_plot)[feature].plot(kind='kde', ax=ax1, label=class_to_plot)

            plt.title(feature)

            df.groupby(class_to_plot)[feature].plot(kind='kde', ax=ax2)
            plt.savefig(output_path + "densityplot_before_after_%s.png"%feature, format='png')
            plt.show()
        except:
            pass


# # Correlation with booking

# In[107]:



def categorical_plot(df, feature, class_to_plot):
    cat_feat = 'cat'
    df_temp=df
    df_temp[cat_feat]=pd.cut(df[feature], right=False, bins=5)
    sp = sns.pointplot(x=cat_feat, y=class_to_plot, data=df_temp)
    plt.xlabel(feature)
    plt.xticks(rotation=70)
    plt.show()
    


#sns.swarmplot(x='prop_starrating', y="booking_bool", data=df)


# In[108]:


# New feature creation
# TODO: move to data_preprocessing file 
# Here to compare before and after
if create_plots:

    class_to_plot = "booking_bool"
    categorical_plot(df,feature , class_to_plot)
    feature = "price_usd_norm_srch_id"
    categorical_plot(df,feature,class_to_plot )

    feature = "log_price_usd"
    df_original[feature] = df[feature]

    class_to_plot = "relevance"
    class_to_plot = "booking_bool"
    categorical_plot(df,feature , class_to_plot)
    feature = "price_usd_log_norm_srch_id"
    categorical_plot(df,feature,class_to_plot )

    feature = "value_for_money"
    feature = "prop_starrating"
    categorical_plot(df,feature,class_to_plot)


# In[109]:

if create_plots:
    feature = "srch_booking_window"
    class_to_plot = "relevance"
    feature = "prop_location_score1"
    categorical_plot(df,feature,class_to_plot)
    feature = "prop_location_score2"
    categorical_plot(df,feature,class_to_plot)
    feature = "prop_location_score_mean"
    categorical_plot(df,feature,class_to_plot)


# In[110]:

# Can we improve this one?
if create_plots:
    feature = "prop_brand_bool"
    categorical_plot(df,feature,class_to_plot)


# In[ ]:




# In[113]:

#create_plots=True
comp_feats=['comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate']

for feat in comp_feats:


    #df[feat + "_percent_diff_signed"]=np.log10(df_original[feat+"_percent_diff"])*df_original[feat]
    #df[feat + "_percent_diff_signed"] = np.log10(df[feat + "_percent_diff_signed"]).replace([np.inf, -np.inf], np.nan)
    if create_plots:
        categorical_plot(df,feat + "_percent_diff_signed",class_to_plot)

    feature =  feat + "_percent_diff_signed"
    #df[feature+"_norm"]=(df[feature] - df[feature].mean()) / (df[feature].std())

    feature = feat + "_percent_diff_signed_norm"
    if create_plots:
        categorical_plot(df,feature,class_to_plot)
    plt.show()
    



# In[112]:


# Take an average of the comp1_rate_percent_diff after normalizing
comp_feats=['comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate']
comp_feats_signed=[]
for feat in comp_feats:
    comp_feats_signed.append(feat+'_percent_diff_signed')
if create_plots:
    df['comp_rate_percent_diff_mean'].hist()
    plt.show()
    categorical_plot(df,'comp_rate_percent_diff_mean',class_to_plot)



# In[40]:


if create_plots:
    feature = "log_price_usd"
    categorical_plot(df,feature,class_to_plot)
    feature="price_usd_norm_srch_id"
    categorical_plot(df,feature,class_to_plot)


# In[114]:

def compare_corr(df, feat1,feat2):
    cor1= np.corrcoef(df[feat1], df['relevance'])[0,1]
    cor2 = np.corrcoef(df[feat2], df['relevance'])[0,1]
    return(cor1,cor2)


try:
    df = df.drop(['cat'], axis=1)
except:
    pass

df = data_preprocessing.missing_values(df, all_cols)


# In[117]:
if training_data:
    compare_corr(df, 'price_usd','price_usd_norm_srch_id')


# In[118]:

try:
    df = df.drop(['cat'], axis=1)
except:
    pass


if training_data:
    all_cols=df.columns.drop(['booking_bool', 'click_bool', 'gross_bookings_usd', 'position'])
else:
    all_cols=df.columns

for feature in all_cols:
    df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
    df[feature][df[feature].isnull()] = df[feature].median()


# ## Prepare for saving cleaned dataset 
# 



# correlated features, they were used to create new ones therefore drop to reduce 
# mutual info calculation 
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
 'comp7_rate_percent_diff_signed_norm']


# In[120]:

for feat in TO_DROP:
    try:
        df = df.drop(feat, axis=1)
    except:
        pass


df1 = df.iloc[:, :35]
df2 = df.iloc[:, 35:]

part = 'tail'
if training_data:
    df1.to_pickle('training_cleaned_dataset_part1'+str(n_select)+part)
    df2.to_pickle('training_cleaned_dataset_part2'+str(n_select)+part)
else:
    df1.to_pickle('test_cleaned_dataset_part1'+str(n_select)+part)
    df2.to_pickle('test_cleaned_dataset_part2'+str(n_select)+part)


overall_time = (time.time() - overall_start_time  )/60
print("Finished processing, took: %.2f minutes", overall_time)
# In[ ]:



