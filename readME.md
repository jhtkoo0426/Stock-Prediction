# Stock Prediction
Predicting stock price trends using time series analysis. This is a learning project to attempt to apply my practical
knowledge in machine learning to financial markets. This project is not designed for production!

---
### Initial results
Using a 4-layer LSTM model consisting of 2 LSTM layers and 2 dense layers, the model was able to make forecasts for the 
shares price of Apple Inc. (AAPL), with a root mean squared error of around USD $1.5.

To put this result into perspective, the RMSE of this model was approximately 0.8% of the highest price of AAPL so far 
(USD $181.26). 

---
### Running the project
1. Clone the repo
2. Open the project with an IDE of your choice
3. Register for an API at https://finnhub.io. This should take less than a minute.
4. Add your API key to your environment variables under the variable name `FINNHUB_API_KEY`. You may have to restart the
IDE for changes to take place.
5. In `src/stock_prediction.py`, enter any US stock exchange symbol as an argument in the `build_and_predict()` function.
6. Run `src/stock_prediction.py`. A graph will be plotted, indicating the training results of the model. The RMSE for 
the model will be printed out as well.

---
### Technologies Used
- Python
- [Finnhub Stock API](https://finnhub.io)
