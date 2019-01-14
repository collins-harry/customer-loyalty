
# coding: utf-8

# In[54]:


import os
import datetime
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
if 'current_dir' in locals():
    print("Directory has been set before:")
else:
    current_dir = os.getcwd()[:-7] 
print(current_dir)
os.chdir(current_dir)
new_test_features = pd.read_csv(current_dir + '/Customer-Loyalty/features/train_feature_fam.csv')
new_train_features = pd.read_csv(current_dir + '/Customer-Loyalty/features/test_feature_fam.csv')


# In[2]:


merchantscsv = pd.read_csv(current_dir + '/Customer-Loyalty/all/merchants.csv')
ht = pd.read_csv(current_dir + '/Customer-Loyalty/all/historical_transactions.csv')


# In[70]:


ht_section = ht
merchantscsv_nodupes = merchantscsv.drop(['merchant_category_id', 'subsector_id', 'category_1', 'city_id', 'state_id', 'category_2'],axis=1)
merchantscsv_section = merchantscsv_nodupes


# In[71]:


ht_merchantscsv_merged = ht_section.merge(merchantscsv_section,how="left",on="merchant_id")


# In[ ]:


ht_merchantscsv_merged.to_csv('/Customer-Loyalty/all/historical_transactions_merchants_merged.csv', index = False)

