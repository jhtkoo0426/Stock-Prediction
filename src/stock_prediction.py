"""
Training a model to predict the price trend for a single stock.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
import time
import datetime
import finnhub as fh
from sklearn.model_selection import train_test_split

FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY')


def plot_series_from_df(df, symbol):
    df.plot()
    plt.title('Share Price of {} from 2000 to now'.format(symbol))
    plt.xlabel('Time')
    plt.ylabel('Share Price')
    plt.show()

# Calculates the timestamps for today and 2000-01-01.
def get_timestamps():
    start = time.mktime(datetime.datetime.strptime('01/01/2000', '%d/%m/%Y').timetuple())
    end = time.time()
    return int(start), int(end)


# Download historical data for one stock symbol from Finnhub.
def download_data_as_df(symbol):
    fh_client = fh.Client(api_key=FINNHUB_API_KEY)
    start, end = get_timestamps()
    data = fh_client.stock_candles(symbol, 'D', start, end)
    df = pd.DataFrame(data)
    df = df.drop(['s', 't', 'v'], axis=1)
    return df


# Prepare dataframe for model training.
def df_to_tensor(df):
    train_df, test_df = train_test_split(df, test_size=0.2)  # Split dataframe into 2 datasets.
    train_ds, test_ds = tf.data.Dataset.from_tensor_slices(dict(train_df)), \
                        tf.data.Dataset.from_tensor_slices(dict(test_df))
    print(len(train_ds), len(test_ds))
    return train_ds, test_ds


data = download_data_as_df("aapl")
plot_series_from_df(data, 'aapl')
