"""
Training a model to predict the price trend for a single stock.
"""
import math
import os

import keras.callbacks
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


# Callback methods for training
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))


# Download historical data for one stock symbol from Finnhub.
def download_stock_data(symbol):
    start = time.mktime(datetime.datetime.strptime('01/01/2000', '%d/%m/%Y').timetuple())
    end = time.time()
    fh_client = fh.Client(api_key=FINNHUB_API_KEY)
    candles_dict = fh_client.stock_candles(symbol, 'D', int(start), int(end))
    stock_df = pd.DataFrame(candles_dict)
    # print(stock_df.shape)
    return stock_df


# Create a new dataframe with only the closing column
def get_closing_column(df):
    col = df.filter(['c'])
    col_arr = col.values       # Convert to np.array
    # print(dataset.shape)
    # print(dataset)
    return col, col_arr


# Calculate the amount of training data
def calc_training_data_len(ds):
    length = math.ceil(len(ds) * .8)
    # print(length)
    return length


def create_scaler():
    min_scaler = MinMaxScaler(feature_range=(0, 1))
    return min_scaler


# Scale the data for training
def rescale(ds, min_scaler):
    rescaled_data = min_scaler.fit_transform(ds)
    # print(rescaled_data)
    return rescaled_data


# Create the training dataset
def build_training_ds(dataset, training_data_len):
    train = dataset[0:training_data_len, :]
    x_train, y_train = [], []       # Split the data into x_train and y_train datasets

    for i in range(60, len(train)):
        x_train.append(train[i - 60:i, 0])  # Contains 60 values
        y_train.append(train[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the x training dataset to fit LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # No. rows, no. of columns, 1
    return x_train, y_train


# Create the testing dataset
def build_testing_ds(dataset, scaled_dataset, length):
    # Create the testing dataset
    testing_dataset = scaled_dataset[length - 60:, :]
    x_testing = []
    y_testing = dataset[length:, :]
    for i in range(60, len(testing_dataset)):
        x_testing.append(testing_dataset[i - 60:i, 0])

    # Convert data to np.array
    x_testing = np.array(x_testing)
    x_testing = np.reshape(x_testing, (x_testing.shape[0], x_testing.shape[1], 1))
    return x_testing, y_testing


# Build and train LSTM model
def run_model(X_train, Y_train):
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model

    # Train the model
    model.fit(X_train, Y_train, batch_size=60, epochs=100, callbacks=[CustomCallback()])
    return model


# Main function
def build_and_predict(symbol):
    df = download_stock_data(symbol)
    data, dataset = get_closing_column(df)
    training_data_len = calc_training_data_len(dataset)

    scaler = create_scaler()
    scaled_data = rescale(dataset, scaler)

    xt, yt = build_training_ds(scaled_data, training_data_len)
    trained_model = run_model(xt, yt)

    x_test, y_test = build_testing_ds(dataset, scaled_data, training_data_len)

    # Get the model's predicted price values
    predictions = trained_model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)         # Un-scaling the transformed values

    # Get the RMSE
    rmse = np.sqrt(np.mean(predictions-y_test)**2)

    metrics = {
        "rmse": rmse,
    }

    return predictions, metrics


def plot_results(data, length, predictions):
    # Prepare results for plot
    train_data_points = data[:length]
    validation_data_points = data[length:]
    validation_data_points['Predictions'] = predictions

    # Plot results
    plt.figure(figsize=(16, 8))
    plt.title('Stock prediction for aapl using LSTM')
    plt.xlabel('Time')
    plt.ylabel('Price of aapl in USD ($)')
    plt.plot(train_data_points['c'])
    plt.plot(validation_data_points[['c', 'Predictions']])
    plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
    plt.grid()
    plt.show()


predictions, metrics = build_and_predict('aapl')
plot_results(data, training_data_len, predictions)