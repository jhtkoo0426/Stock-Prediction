"""
Training a model to predict the price trend for a single stock.
"""
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import datetime
import finnhub as fh
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense


FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY')


# Download historical data for one stock symbol from Finnhub.
start = time.mktime(datetime.datetime.strptime('01/01/2000', '%d/%m/%Y').timetuple())
end = time.time()
fh_client = fh.Client(api_key=FINNHUB_API_KEY)
candles_dict = fh_client.stock_candles('aapl', 'D', int(start), int(end))
df = pd.DataFrame(candles_dict)
print(df)
print(df.shape)


# Visualise closing price history
plt.figure()
plt.title('Closing price history of aapl')
plt.plot(df['c'])
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.show()


# Create a new dataframe with only the closing column
data = df.filter(['c'])
dataset = data.values       # Convert to np.array
print(dataset.shape)
training_data_len = math.ceil(len(dataset) * .8)
print(training_data_len)


# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)


# Create the training dataset
training_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []       # Split the data into x_train and y_train datasets

for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])        # Contains 60 values
    y_train.append(training_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()


# Convert training datasets to np arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the x training dataset to fit LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))        # No. rows, no. of columns, 1
print(x_train.shape)

