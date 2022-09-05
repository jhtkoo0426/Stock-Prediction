"""
Training a model to predict the price trend for a single stock.
"""
import math
import os

import finnhub
import keras.callbacks
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import finnhub as fh
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense


FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY')


# Callback methods
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))


class StockPrediction:
    def __init__(self):
        self.api_key = None
        self.finnhub_client = None
        self.min_max_scaler = None
        self.scaled_price_values = None
        self.training_data_length = None
        self.x_training = None
        self.y_training = None
        self.x_testing = None
        self.y_testing = None
        self.trained_model = None

    def add_api_key(self, api_key: str):
        self.api_key = os.environ.get(api_key)
        self.finnhub_client = finnhub.Client(api_key=api_key)

    def download_stock_data(self, symbol: str):
        start = time.mktime(
            datetime.datetime.strptime('01/01/2000', '%d/%m/%Y').timetuple()        # Timestamp of 1 Jan, 2000.
        )
        end = time.time()   # Current timestamp.

        candles = self.finnhub_client.stock_candles(symbol, 'D', int(start), int(end))
        candlesDataframe = pd.DataFrame(candles)
        return candlesDataframe

    @staticmethod
    def get_closing_price_column(dataframe):
        column = dataframe.filter(['c'])
        return column, column.values

    def get_training_data_length(self, dataframe):
        self.training_data_length = math.ceil(len(dataframe) * 0.8)

    def create_scaler(self):
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))

    def rescale_dataframe(self, dataframe):
        self.scaled_price_values = self.min_max_scaler.fit_transform(dataframe)

    def build_training_dataset(self):
        trainingDataset = self.scaled_price_values[0:self.training_data_length, :]
        self.x_training, self.y_training = [], []  # Split the data into x_train and y_train datasets

        for i in range(60, len(trainingDataset)):
            self.x_training.append(trainingDataset[i - 60:i, 0])
            self.y_training.append(trainingDataset[i, 0])

        self.x_training, self.y_training = np.array(self.x_training), np.array(self.y_training)

        # Reshape the x training dataset to fit LSTM model
        self.x_training = np.reshape(self.x_training, (self.x_training.shape[0], self.x_training.shape[1], 1))  # No. rows, no. of columns, 1

    def build_testing_dataset(self, unscaledDataframe):
        testingDataset = self.scaled_price_values[self.training_data_length - 60:, :]
        self.x_testing, self.y_testing = [], unscaledDataframe[self.training_data_length:, :]

        for i in range(60, len(testingDataset)):
            self.x_testing.append(testingDataset[i - 60:i, 0])

        self.x_testing = np.array(self.x_testing)
        self.x_testing = np.reshape(self.x_testing, (self.x_testing.shape[0], self.x_testing.shape[1], 1))

    def train_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.x_training.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.x_training, self.y_training, batch_size=1, epochs=1, callbacks=[])
        return model

    def build(self, symbol: str):
        candlesDataframe = self.download_stock_data(symbol)
        self.closingPriceColumn, unscaledPriceValues = self.get_closing_price_column(candlesDataframe)
        self.get_training_data_length(unscaledPriceValues)

        self.create_scaler()
        self.rescale_dataframe(self.closingPriceColumn)

        self.build_training_dataset()
        self.build_testing_dataset(unscaledPriceValues)

        self.trained_model = self.train_model()

    def predictTrend(self):
        predictions = self.trained_model.predict(self.x_testing)
        predictions = self.min_max_scaler.inverse_transform(predictions)

        metrics = {
            "rmse": np.sqrt(np.mean(predictions - self.y_testing)**2)
        }
        return predictions, metrics

    def plot(self, symbol, model_predictions):
        # Prepare results for plot
        trainDataPoints = self.closingPriceColumn[:self.training_data_length]
        validationDataPoints = self.closingPriceColumn[self.training_data_length:]
        validationDataPoints['Predictions'] = model_predictions

        # Plot results
        plt.figure(figsize=(16, 8))
        plt.title(f'Stock prediction of {symbol} using LSTM')
        plt.xlabel('Time')
        plt.ylabel(f'Price of {symbol} in USD ($)')
        plt.plot(trainDataPoints['c'])
        plt.plot(validationDataPoints[['c', 'Predictions']])
        plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
        plt.grid()
        plt.show()


sym = 'aapl'
fortuneTeller = StockPrediction()
fortuneTeller.add_api_key(FINNHUB_API_KEY)
fortuneTeller.build(sym)
p, m = fortuneTeller.predictTrend()
fortuneTeller.plot(sym, p)
print(f"RMSE for the model was {m['rmse']}")