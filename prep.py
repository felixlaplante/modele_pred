import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import config

def load_date_df(df):
    date_cont_df = df[config.cont_date_cols]
    date_cat_df = pd.get_dummies(df[config.cat_date_cols])
    
    date_df = pd.DataFrame(
        data=np.hstack([date_cont_df.values, date_cat_df.values]),
        columns=config.cont_date_cols + list(date_cat_df.columns),
        index=df.index)

    return date_df
    

def load_weather_df(df):
    return df[config.weather_cols]


def load_fr_co_emissions_df(codf, df):
    fr_co_emissions_df = codf[codf['Entity'] == 'France'].copy()
    fr_co_emissions_df = fr_co_emissions_df.drop(columns='Entity')
    fr_co_emissions_df = fr_co_emissions_df.resample('D').interpolate(method='cubic', limit_direction='both')
    fr_co_emissions_df = fr_co_emissions_df.reindex(df.index)
    
    return fr_co_emissions_df


def load_lag_df(df):
    return df[config.lag_cols]


def load_seq(X, y):
    X_seq = []
    y_seq = None if y is None else y[config.seq_length-1:]
    
    for i in range(X.shape[0] - config.seq_length + 1):
        seq = X[i:i+config.seq_length]
        X_seq.append(seq)

    return np.array(X_seq), y_seq


def get_scalers(X, y):
    continuous_cols_idxs = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 2]

    X_scaler= ColumnTransformer([('scaler', StandardScaler(), continuous_cols_idxs)], remainder='passthrough').fit(X)
    y_scaler = StandardScaler().fit(y)
    return X_scaler, y_scaler  