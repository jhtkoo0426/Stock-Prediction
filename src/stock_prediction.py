"""
Training a model to predict the price trend for a single stock.
"""
import math
import os
import keras.callbacks
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import finnhub as fh
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Callback methods
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))


# Class responsible for establishing connections to the Finnhub API to fetch data.
class FinnhubClient:
    def __init__(self):
        """
        Construct a new 'FinnhubClient' instance.

        :return: returns nothing
        """

        self.api_key = None
        self.client = None

    def update_key(self, key: str):
        """
        Updates the current key with a new key. If no key was previously set, use the key as the current key.

        :param key: a registered API key from Finnhub
        :return: returns nothing
        """

        self.api_key = key

    def create_client(self):
        """
        Establish a connection to the Finnhub API using the current api key

        :return: returns nothing
        """

        self.client = fh.Client(api_key=self.api_key)

    def download_stock_data(self, symbol: str):
        """
        Fetches historical share price data of a stock starting from 1 Jan, 2000.

        :param symbol: an abbreviation used to uniquely identify publicly traded shares of a particular stock on a
                       particular stock market
        :return: a dataframe consisting of open, close, high, low share prices and the volume of trades of the day
        """

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

        self.trainingDataX, self.trainingDataY = [], []
        self.testingDataX, self.testingDataY = [], None

        self.model = None

    def create_processor(self, dataframe: pd.DataFrame):
        """
        Initialize the data processor with a dataframe. Also re-indexes the dataframe to a timestamp column.

        :param dataframe: the target dataframe for processing
        :return: returns nothing
        """
        dataframe['t'] = [datetime.datetime.fromtimestamp(x) for x in dataframe['t']]
        dataframe.set_index('t', inplace=True)
        self.raw_dataframe = dataframe

    def set_training_data_length(self, split_percentage: float):
        """
        Set the amount of training data as a proportion of the full dataset.

        :param split_percentage: a percentage float
        :return: returns nothing
        """
        if self.raw_dataframe is not None:
            self.training_data_length = math.ceil(len(self.raw_dataframe) * split_percentage)

    def get_column(self, column_names: list):
        """

        :param column_names: a list of column names
        :return: a filtered dataframe consisting of the selected columns by name only
        """

        return self.raw_dataframe.filter(column_names)

    def set_unscaled_prices(self):
        """
        Create a list consisting of the closing price values of a symbol. The values must be unscaled.

        :return: returns nothing
        """

        self.unscaled_prices = self.get_column(column_names=['c']).values

    def rescale_prices(self):
        """
        Rescale the closing prices for training.

        :return: returns nothing
        """

        closingPriceColumn = self.get_column(column_names=['c'])
        self.scaled_prices = self.scaler.fit_transform(closingPriceColumn)

    @staticmethod
    def reshape_array(np_array):
        """
        Reshape an array to fit the LSTM model's shape.

        :param np_array: a numpy array
        :return: a reshaped numpy array with the same dimensions as the LSTM model
        """

        return np.reshape(np_array, (np_array.shape[0], np_array.shape[1], 1))

    def build_training_dataset(self):
        """
        Create a training dataset from the raw dataframe.

        :return: returns nothing
        """

        trainingDataset = self.scaled_prices[0:self.training_data_length, :]        # Split the dataframe for training

        for i in range(60, len(trainingDataset)):
            self.trainingDataX.append(trainingDataset[i-60:i, 0])
            self.trainingDataY.append(trainingDataset[i, 0])

        self.trainingDataX = self.reshape_array(np.array(self.trainingDataX))
        self.trainingDataY = np.array(self.trainingDataY)

    def build_testing_dataset(self):
        """
        Create a testing dataset from the raw dataframe.

        :return: returns nothing
        """

        testingDataset = self.scaled_prices[self.training_data_length - 60: , :]    # Split the dataframe for testing
        self.testingDataX = []

        for i in range(60, len(testingDataset)):
            self.testingDataX.append(testingDataset[i-60:i, 0])

        self.testingDataX = self.reshape_array(np.array(self.testingDataX))
        self.testingDataY = self.unscaled_prices[self.training_data_length:, :]

    def train_model(self, batch_size: int, epochs: int):
        """
        Train a LSTM model.

        :param batch_size: an integer to indicate the number of training examples to be used in one epoch
        :param epochs: an integer to indicate the number of epochs used to train the model.
        :return: returns nothing.
        """

        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.trainingDataX.shape[1], 1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.trainingDataX, self.trainingDataY, batch_size=batch_size, epochs=epochs, callbacks=[])

    def predict_against_validation(self):
        """
        Make predictions against the validation dataset using a trained LSTM model.

        :return: predictions: a list of predicted share price values generated from the model;
                 metrics: a dictionary of various metrics
        """
        predictions = self.model.predict(self.testingDataX)
        predictions = self.scaler.inverse_transform(predictions)

        metrics = {
            "rmse": np.sqrt(np.mean(predictions - self.testingDataY) ** 2)
        }
        return predictions, metrics

    def predict_new_prices(self, days: int):
        # Use the last n prices to predict future share prices.
        predictions = self.model.predict(self.testingDataX[-days:])
        predictions = self.scaler.inverse_transform(predictions)
        return predictions


class StockPrediction:
    def __init__(self, symbol, api_key):
        """
        The main class to combine everything together.

        :param symbol: an abbreviation used to uniquely identify publicly traded shares of a particular stock on a
                       particular stock market
        :param api_key: a registered API key from Finnhub
        """

        self.symbol = symbol
        self.api_key = api_key
        self.trained_model = None

        self.finnhub_client = FinnhubClient()
        self.finnhub_client.update_key(api_key)
        self.finnhub_client.create_client()

        self.processor = StockDataProcessor()

    def create_data_processor(self):
        candles = self.finnhub_client.download_stock_data(self.symbol)
        self.processor.create_processor(candles)
        self.processor.set_training_data_length(0.8)
        self.processor.set_unscaled_prices()
        self.processor.rescale_prices()
        self.processor.build_training_dataset()
        self.processor.build_testing_dataset()

    def trainModel(self, batch_size, epochs):
        self.processor.train_model(batch_size=batch_size, epochs=epochs)

    def predict_validation_set(self):
        predictions, metrics = self.processor.predict_against_validation()
        return predictions, metrics

    def predict_future(self, days):
        return self.processor.predict_new_prices(days)

    def plot_validation_results(self, predictions):
        """
        Plot a graph of the real historical share price data and the model's predictions against time.

        :param predictions: predictions generated from the model
        :return: returns nothing
        """
        closingPricesColumn = self.processor.get_column(['c'])
        trainingDataPoints = closingPricesColumn[:self.processor.training_data_length]
        validationDataPoints = closingPricesColumn[self.processor.training_data_length:]
        try:
            validationDataPoints['Predictions'] = predictions
        except SettingWithCopyWarning:
            pass

        # Plot results
        plt.figure(figsize=(16, 8))
        plt.title(f'Stock prediction of {self.symbol} against validation set')
        plt.xlabel('Time')
        plt.ylabel(f'Price of {self.symbol} in USD ($)')
        plt.plot(trainingDataPoints['c'])
        plt.plot(validationDataPoints[['c', 'Predictions']])
        plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
        plt.grid()
        plt.show()

    def plot_future_results(self, predictions):
        closingPricesColumn = self.processor.get_column(['c'])
        print(closingPricesColumn)
        predictions = pd.DataFrame(predictions)
        predictions.index += len(self.processor.raw_dataframe)
        print(predictions)

        plt.figure(figsize=(16, 8))
        plt.title(f'Stock forecasting of {self.symbol}')
        plt.xlabel('Time')
        plt.ylabel(f'Price of {self.symbol} in USD ($)')
        plt.plot(closingPricesColumn)
        plt.plot(predictions)
        plt.legend(['Past prices', 'Future prices'], loc='lower right')
        plt.grid()
        plt.show()


sym = 'aapl'

predictor = StockPrediction(sym, os.environ.get('FINNHUB_API_KEY'))
predictor.create_data_processor()
predictor.trainModel(batch_size=60, epochs=100)

p, m = predictor.predict_validation_set()
predictor.plot_validation_results(p)
print(m)

future = predictor.predict_future(30)
predictor.plot_future_results(future)