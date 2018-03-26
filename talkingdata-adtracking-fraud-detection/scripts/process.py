
# coding: utf-8

# In[1]:


#对一些app如9,19,10等，hour 6, 11, 15有问题，可以考虑单独对19建模
#感觉时间序列分析是关键，双11行为变化难以预测


# In[2]:


import gc
import numpy as np
import pandas as pd
import lightgbm
from sklearn import model_selection


# In[3]:


k = 23333333
train_file_in = '../data/train.csv'
sub_file_in = '../data/test.csv'


# In[4]:


dtypes = {
'ip'            : 'uint32',
'app'           : 'uint16',
'device'        : 'uint16',
'os'            : 'uint16',
'channel'       : 'uint16',
'is_attributed' : 'uint8',
'click_id'      : 'uint32'
}
df = pd.read_csv(train_file_in, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
df_sub = pd.read_csv(sub_file_in, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])


# In[5]:


len_train = df.shape[0]
df = df.append(df_sub)


# In[6]:


print('add feature hour')
df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')


# In[7]:


print('grouping by ip-day-hour, count channel')
gp = df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_hour_count'})
print("merging...")
df = df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()


# In[8]:


print('group by ip-app, count channel')
gp = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
print("merging...")
df = df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()


# In[9]:


print('group by ip-app-os, count channel')
gp = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
print("merging...")
df = df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()


# In[10]:


df_train = df[: len_train - k]
df_val = df[len_train - k: len_train]
df_sub = df[len_train: ]
del df
gc.collect()


# In[11]:


xnames = ['app', 'device', 'os', 'channel', 'hour', 'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count']
yname = 'is_attributed'
categorical_names = ['app', 'device', 'os', 'channel', 'hour'] 


# In[12]:


params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 5,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 99, # because training data is extremely unbalanced 
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
    }


# In[13]:


data_val = lightgbm.Dataset(df_val[xnames], df_val[yname], categorical_feature=categorical_names)
data = lightgbm.Dataset(df_train[xnames], df_train[yname], categorical_feature=categorical_names)


# In[14]:


evals_result={}
r = lightgbm.train(params,
                   data,
                   valid_sets=[data, data_val],
                   valid_names=['train', 'valid'], 
                   evals_result=evals_result,
                   num_boost_round=300,
                   early_stopping_rounds=30
                  )


# In[11]:


#0.972 ip no category


# In[15]:


ids = df_sub['click_id'].values
X_sub = df_sub[xnames].values


# In[16]:


y_sub = r.predict(X_sub)


# In[19]:


output = pd.DataFrame({'click_id': ids, 'is_attributed': y_sub})
output.to_csv("../data/submission.csv", index=False)

