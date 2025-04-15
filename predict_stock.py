import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model # type: ignore
import joblib

def get_latest_stock_data(stock_symbol='AAPL', window_size=60):
    data = yf.download(stock_symbol, period='90d')  # get last 3 months just in case
    close_prices = data[['Close']]

    scaler = joblib.load('model/scaler.pkl')
    scaled_data = scaler.transform(close_prices)

    last_window = scaled_data[-window_size:]
    X_input = np.reshape(last_window, (1, window_size, 1))

    return X_input, scaler, close_prices

def make_prediction():
    model = load_model('model/lstm_model.h5')
    X_input, scaler, close_prices = get_latest_stock_data()

    prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction).flatten()

    # 📊 Plot results
    last_actual = close_prices[-1:].values.flatten()[0]
    future_days = [f'Day {i+1}' for i in range(5)]

    plt.figure(figsize=(8, 5))
    plt.plot(future_days, prediction, marker='o', label='Predicted Prices', color='blue')
    plt.axhline(last_actual, color='gray', linestyle='--', label='Last Actual Price')
    plt.title('📈 Next 5-Day Stock Price Prediction')
    plt.xlabel('Future Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

make_prediction()
