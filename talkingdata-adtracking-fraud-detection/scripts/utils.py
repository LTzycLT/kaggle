import gc

xnames = ['app', 'device', 'os', 'channel', 'hour', 'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count']
yname = 'is_attributed'
categorical_names = ['app', 'device', 'os', 'channel', 'hour']

def load(len_train=184903890, k = 23333333):
    import pickle
    with open('../data/features.pickle', 'rb') as handle:
        df = pickle.load(handle)
    df_train = df[: len_train - k]
    df_val = df[len_train - k: len_train]
    df_sub = df[len_train: ]
    print('df_train.shape %s,%s' % df_train.shape)
    print('df_val.shape %s,%s' % df_val.shape)
    print('df_sub.shape %s,%s' % df_sub.shape)
    del df
    gc.collect()
    return df_train, df_val, df_sub
