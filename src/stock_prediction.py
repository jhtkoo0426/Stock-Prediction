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


# Callback methods for training
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))


# Downloads dataframe of stock symbol from Finnhub. Data includes open, high, low, close prices and volume of the stock.
def downloadStockData(symbol):
    start = time.mktime(datetime.datetime.strptime('01/01/2000', '%d/%m/%Y').timetuple())
    end = time.time()

    finnhubClient = fh.Client(api_key=FINNHUB_API_KEY)
    candlesDict = finnhubClient.stock_candles(symbol, 'D', int(start), int(end))
    stockDataframe = pd.DataFrame(candlesDict)
    return stockDataframe


# Create a new dataframe with only the closing column
def getClosingPriceColumn(dataframe):
    col = dataframe.filter(['c'])
    col_arr = col.values       # Convert to np.array
    return col, col_arr


# Calculate the amount of training data examples
def calculateTrainingDataLength(dataframe):
    return math.ceil(len(dataframe) * .8)


def createScaler():
    return MinMaxScaler(feature_range=(0, 1))


# Scale the data for training
def rescaleData(scaler, dataframe):
    return scaler.fit_transform(dataframe)


# Create the training dataset
def buildTrainingDataset(dataset, trainingDataLength):
    trainingDataset = dataset[0:trainingDataLength, :]
    x_train, y_train = [], []                           # Split the data into x_train and y_train datasets

    for i in range(60, len(trainingDataset)):
        x_train.append(trainingDataset[i - 60:i, 0])
        y_train.append(trainingDataset[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the x training dataset to fit LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # No. rows, no. of columns, 1
    return x_train, y_train


# Create the testing dataset
def buildTestingDataset(dataset, scaledDataset, trainingDataLength):
    testingDataset = scaledDataset[trainingDataLength - 60:, :]        # Create the testing dataset
    x_testing, y_testing = [], dataset[trainingDataLength:, :]

    for i in range(60, len(testingDataset)):
        x_testing.append(testingDataset[i - 60:i, 0])

    x_testing = np.array(x_testing)
    x_testing = np.reshape(x_testing, (x_testing.shape[0], x_testing.shape[1], 1))
    return x_testing, y_testing


# Build and train LSTM model
def trainModel(x_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=60, epochs=100, callbacks=[CustomCallback()])
    return model


def plot_results(data, symbol, length, modelPredictions):
    # Prepare results for plot
    trainDataPoints = data[:length]
    validationDataPoints = data[length:]
    validationDataPoints['Predictions'] = modelPredictions

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


# Create training and testing datasets
def buildTrainModel(symbol):
    stockCandlesDataframe = downloadStockData(symbol)
    closingPriceColumn, unscaledPriceValues = getClosingPriceColumn(stockCandlesDataframe)

    trainingDataLength = calculateTrainingDataLength(unscaledPriceValues)

    scaler = createScaler()
    scaledPriceValues = rescaleData(scaler, unscaledPriceValues)
    x_training, y_training = buildTrainingDataset(scaledPriceValues, trainingDataLength)
    x_testing, y_testing = buildTestingDataset(unscaledPriceValues, scaledPriceValues, trainingDataLength)

    trainedModel = trainModel(x_training, y_training)
    modelParams = {
        "closingPriceColumn": closingPriceColumn,       # Used to plot graph
        "scaler": scaler,                               # Used to inverse transform predictions into symbol share prices
        "trainingDataLength": trainingDataLength,
        "x_testing": x_testing,
        "y_testing": y_testing,
    }
    return trainedModel, modelParams


def predictWithModel(model, modelParams):
    scaler, x_testing, y_testing = modelParams["scaler"], modelParams["x_testing"], modelParams["y_testing"]
    predictions = model.predict(x_testing)
    predictions = scaler.inverse_transform(predictions)

    metrics = {
        "rmse": np.sqrt(np.mean(predictions - y_testing) ** 2)
    }

    return predictions, metrics

# def saveModel(model):
#     model.save()


sym = 'aapl'
testModel, testModelParams = buildTrainModel(sym)
dataForPlot = testModelParams["closingPriceColumn"]
tdl = testModelParams["trainingDataLength"]
predict, metric = predictWithModel(testModel, testModelParams)
testModelParams["symbol"] = sym
plot_results(dataForPlot, sym, tdl, predict)
print(f"RMSE for the model was: {metric['rmse']}")