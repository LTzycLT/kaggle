
# coding: utf-8

# In[4]:

from __future__ import print_function
import pandas as pd
import numpy as np


# In[5]:

df = pd.read_csv('./input/train_ver2.csv')

#df_train = df[df.fecha_dato != '2016-05-28']
#df_val = df[df.fecha_dato == '2016-05-28']

cnames = list(df)


# In[6]:

df = df.drop_duplicates(subset=['ncodpers'], keep='last')


# In[7]:

preds = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
fs =  ['ncodpers', 'fecha_dato', 'sexo', 'age', 'ind_actividad_cliente', 'renta', 'segmento']#, 'antiguedad'


# In[8]:

df_fecha_dato = pd.get_dummies(df['fecha_dato'])
df_sexo = pd.get_dummies(df['sexo'])
df_age = pd.get_dummies(df['age'])
df_segmento = pd.get_dummies(df['segmento'])
features = pd.concat([df[fs + preds].drop(['fecha_dato', 'sexo', 'age', 'segmento'], axis=1), df_fecha_dato, df_sexo, df_age, df_segmento], axis=1)


# In[10]:

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import defaultdict
#from xgboost.sklearn import XGBModel

res = defaultdict(list)

features.fillna(0, inplace=True)
for c in preds:
    if c == 'ind_cco_fin_ult1': break
    model = LogisticRegression()
    ids = features['ncodpers'].astype(int).values
    X = features.drop(['ncodpers', c], axis = 1).astype(float).values
    y = features[c].astype(int).values
    std = StandardScaler()
    X = std.fit_transform(X)
    model.fit(X, y)
    y_proba = model.predict_proba(X)[:, 1]
    for i in range(ids.shape[0]):
        if y[i] == 0: res[ids[i]].append((c, y_proba[i]))
    print("%s\t%", (c, roc_auc_score(y, y_proba)))


# In[35]:

import csv
df_out = pd.read_csv('./input/test_ver2.csv')[['ncodpers']]
ids = df_out.values.reshape(-1)

added_products = []
for d in ids:
    if d in res: added_products.append(' '.join([x[0] for x in sorted(res[d], key=lambda obj: obj[1], reverse=True)[0: 7]]))
    else: added_products.append("")
df_out['added_products'] = added_products
df_out.to_csv('output/out', index=False, quoting=csv.QUOTE_NONE)


# In[ ]:



