"""
Training a model to predict the price trend for a single stock.
"""
import math
import os

import keras.callbacks
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


# Class responsible for establishing connections to the Finnhub API to fetch data.
class FinnhubClient:
    def __init__(self):
        self.api_key = None
        self.client = None

    def update_key(self, key: str):
        self.api_key = key

    def start_client(self):
        self.client = fh.Client(api_key=self.api_key)

    def download_stock_data(self, symbol: str):
        start = time.mktime(
            datetime.datetime.strptime('01/01/2000', '%d/%m/%Y').timetuple()  # Timestamp of 1 Jan, 2000.
        )
        end = time.time()  # Current timestamp.

        candles = self.client.stock_candles(symbol, 'D', int(start), int(end))
        candlesDataframe = pd.DataFrame(candles)
        return candlesDataframe


# Class responsible for processing ML data
class StockDataProcessor:
    def __init__(self):
        self.raw_dataframe = None
        self.scaled_prices = None
        self.unscaled_prices = None

        self.training_data_length = 0
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.trainingDataX, self.trainingDataY = None, None
        self.testingDataX, self.testingDataY = None, None

        self.model = None

    def create_processor(self, dataframe):
        self.raw_dataframe = dataframe

    def set_training_data_length(self, split_percentage):
        if self.raw_dataframe is not None:
            self.training_data_length = math.ceil(len(self.raw_dataframe) * split_percentage)

    def get_column(self, column_names):
        return self.raw_dataframe.filter(column_names)

    def set_unscaled_prices(self):
        self.unscaled_prices = self.get_column(column_names=['c']).values

    def rescale_prices(self):
        closingPriceColumn = self.get_column(column_names=['c'])
        self.scaled_prices = self.scaler.fit_transform(closingPriceColumn)

    def reshape_array(self, np_array):
        return np.reshape(np_array, (np_array.shape[0], np_array.shape[1], 1))

    def build_training_dataset(self):
        trainingDataset = self.scaled_prices[0:self.training_data_length, :]
        x_train, y_train = [], []

        for i in range(60, len(trainingDataset)):
            x_train.append(trainingDataset[i-60:i, 0])
            y_train.append(trainingDataset[i, 0])

        self.trainingDataX = self.reshape_array(np.array(x_train))
        self.trainingDataY = np.array(y_train)

    def build_testing_dataset(self):
        testingDataset = self.scaled_prices[self.training_data_length - 60: , :]
        x_test = []
        y_test = self.unscaled_prices[self.training_data_length:, :]

        for i in range(60, len(testingDataset)):
            x_test.append(testingDataset[i-60:i, 0])

        self.testingDataX = self.reshape_array(np.array(x_test))
        self.testingDataY = y_test

    def train_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.trainingDataX.shape[1], 1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.trainingDataX, self.trainingDataY, batch_size=1, epochs=1, callbacks=[])

    def predict_with_model(self):
        predictions = self.model.predict(self.testingDataX)
        predictions = self.scaler.inverse_transform(predictions)

        metrics = {
            "rmse": np.sqrt(np.mean(predictions - self.testingDataY) ** 2)
        }
        return predictions, metrics


class StockPrediction:
    def __init__(self, symbol):
        self.finnhub_client = FinnhubClient()
        self.processor = StockDataProcessor()

        self.symbol = symbol
        self.trained_model = None

    def create_finnhub_client(self, api_key: str):
        self.finnhub_client.update_key(api_key)
        self.finnhub_client.start_client()

    def create_data_processor(self):
        candles = self.finnhub_client.download_stock_data(self.symbol)
        self.processor.create_processor(candles)
        self.processor.set_training_data_length(0.8)
        self.processor.set_unscaled_prices()
        self.processor.rescale_prices()
        self.processor.build_training_dataset()
        self.processor.build_testing_dataset()

    def trainModel(self):
        self.processor.train_model()

    def predict(self):
        predictions, metrics = self.processor.predict_with_model()
        return predictions, metrics

    def plot_model_results(self, predictions):
        closingPricesColumn = self.processor.get_column(['c'])
        trainingDataPoints = closingPricesColumn[:self.processor.training_data_length]
        validationDataPoints = closingPricesColumn[self.processor.training_data_length:]
        validationDataPoints['Predictions'] = predictions

        # Plot results
        plt.figure(figsize=(16, 8))
        plt.title(f'Stock prediction of {self.symbol} using LSTM')
        plt.xlabel('Time')
        plt.ylabel(f'Price of {self.symbol} in USD ($)')
        plt.plot(trainingDataPoints['c'])
        plt.plot(validationDataPoints[['c', 'Predictions']])
        plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
        plt.grid()
        plt.show()


sym = 'aapl'

predictor = StockPrediction(sym)
predictor.create_finnhub_client(os.environ.get('FINNHUB_API_KEY'))
predictor.create_data_processor()
predictor.trainModel()

p, m = predictor.predict()
predictor.plot_model_results(p)