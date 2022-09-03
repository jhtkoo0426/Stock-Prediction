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


# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')          # Compile the model


# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# Create the testing dataset
testing_dataset = scaled_data[training_data_len-60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(testing_dataset)):
    x_test.append(testing_dataset[i-60:i, 0])


# Convert data to np.array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)         # Un-scaling the transformed values


# Get the RMSE
rmse = np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)


# Prepare results for plot
train_data_points = data[:training_data_len]
validation_data_points = data[training_data_len:]
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


# Visualising results via terminal
print("Validation results")
print(validation_data_points)