
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

df = pd.read_csv('train_dat', delimiter='\t')


# In[5]:

X = df['goods_id'].values
X = X.reshape((X.shape[0], 1))
y = df['label'].values


# In[6]:

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X = enc.fit_transform(X)  


# In[7]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)


# In[8]:

from  sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


# In[13]:

y_prob = model.predict_proba(X_test)


# In[19]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])


# In[20]:

roc_auc_score(y_test, y_prob[:, 1])


# In[ ]:



