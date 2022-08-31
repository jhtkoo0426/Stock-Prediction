"""
Scrapping historical price data of various technology stocks
"""
import yfinance as yf
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Download & prepare historical data for a single stock symbol.
def prepare_stock_data(symbol):
    stock_df = yf.download(symbol.upper(), period="10y", interval="1d", threads=True)
    train_dataset, test_dataset = train_test_split(stock_df, test_size=0.2)  # pandas.dataframe

    # Convert the datasets into tensors for training & testing.
    train_tensor, test_tensor = tf.data.Dataset.from_tensor_slices(dict(train_dataset)), \
                                tf.data.Dataset.from_tensor_slices(dict(train_dataset))
    return train_tensor, test_tensor


aapl_train, aapl_test = prepare_stock_data("aapl")
print(aapl_train)
