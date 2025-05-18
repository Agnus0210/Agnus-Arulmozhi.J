# Agnus-Arulmozhi.J import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Step 1: Load stock data
def load_data(ticker='AAPL', period='5y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    return data[['Close']]

# Step 2: Preprocess data
def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, time_steps, features]
    
    return X, y, scaler

# Step 3: Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train and predict
def train_and_predict(ticker='AAPL'):
    data = load_data(ticker)
    X, y, scaler = preprocess_data(data)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(y_test_scaled, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Run the pipeline
train_and_predict('AAPL')  # You can change to 'GOOG', 'MSFT', etc.
