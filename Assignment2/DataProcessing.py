
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

training_data = False
select_n_rows = True
head_set=True
n_select =None # this changes downstream if select_n_rows is True

if training_data:
    data_original = pd.read_csv('./Data Mining VU data/training_set_VU_DM_2014.csv')
else:
    data_original = pd.read_csv('./Data Mining VU data/test_set_VU_DM_2014.csv')

# In[88]:

len(data_original)


# In[89]:
if select_n_rows:
    if head_set:
    	n_select = 4000000
    	df = data_original.iloc[:n_select]
    	df_original = data_original.iloc[:n_select]
    else:
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
    all_cols=df.columns.drop(['booking_bool', 'click_bool', 'gross_bookings_usd', 'position', 'srch_id', 'prop_id', 'relevance'])
else:
    all_cols=df.columns.drop(['srch_id', 'prop_id'])


df = data_preprocessing.missing_values(df, all_cols)


# ### Outliers
# For features with high value outliers cap to maximum


# If testing feature engineering use balance_dataset, otherwise use the whole 
# dataset
#df = utils.balance_dataset(df, 3)
df = data_preprocessing.remove_outliers(df, all_cols)

create_plots=None





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





def categorical_plot(df, feature, class_to_plot):
    cat_feat = 'cat'
    df_temp=df
    df_temp[cat_feat]=pd.cut(df[feature], right=False, bins=5)
    sp = sns.pointplot(x=cat_feat, y=class_to_plot, data=df_temp)
    plt.xlabel(feature)
    plt.xticks(rotation=70)
    plt.show()
    




try:
    df = df.drop(['cat'], axis=1)
except:
    pass

df = data_preprocessing.missing_values(df, all_cols)


try:
    df = df.drop(['cat'], axis=1)
except:
    pass



if training_data:
    all_cols=df.columns.drop(['booking_bool', 'click_bool', 'gross_bookings_usd', 'position', 'srch_id', 'prop_id', 'relevance'])
else:
    all_cols=df.columns.drop(['srch_id', 'prop_id'])

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

if head_set:
    part = 'head'
else:
    part='tail'

if training_data:
    df1.to_pickle('training_cleaned_dataset_part1_fix'+str(n_select)+part)
    df2.to_pickle('training_cleaned_dataset_part2_fix'+str(n_select)+part)
else:
    df1.to_pickle('test_cleaned_dataset_part1_fix'+str(n_select)+part)
    df2.to_pickle('test_cleaned_dataset_part2_fix'+str(n_select)+part)


overall_time = (time.time() - overall_start_time  )/60
print("Finished processing, took: %.2f minutes", overall_time)
# In[ ]:



