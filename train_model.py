import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

import joblib

def get_stock_data(stock_symbol='AAPL', period='5y'):
    data = yf.download(stock_symbol, period=period)
    return data[['Close']]

def prepare_data(data, window_size=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data) - 5):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i:i + 5, 0])  # Predict next 5 days

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def train_model():
    data = get_stock_data('AAPL')
    X, y, scaler = prepare_data(data)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(5))  # Predict 5 days
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)

    model.save('model/lstm_model.h5')
    joblib.dump(scaler, 'model/scaler.pkl')
    print("✅ Model and scaler saved!")

train_model()
