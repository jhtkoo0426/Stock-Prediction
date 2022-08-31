"""
Scrapping historical price data of various technology stocks
"""
import yfinance as yf
from sklearn.model_selection import train_test_split


# Download pandas.dataframe for data
def download_stock_data(symbol):
    stock_df = yf.download(symbol, period="10y", interval="1d", threads=True)
    train_dataset, test_dataset = train_test_split(stock_df, test_size=0.2)

    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset