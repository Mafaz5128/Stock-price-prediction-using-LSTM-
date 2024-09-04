import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Define functions for loading data, training model, and making predictions
def load_data(ticker):
    try:
        data = yf.download(tickers=ticker, period='3mo', interval='1h')
        data['Direction'] = np.where(data['Close'] > data['Open'], 1, 0)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_data(data):
    close_price = data['Close'].values
    close_price = close_price.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_price)
    return scaled_data, scaler

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model(time_step):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    st.title('Stock Price Prediction with LSTM')
    
    ticker = st.text_input('Enter Stock Ticker:', 'AMZN')
    data = load_data(ticker)
    
    if not data.empty:
        scaled_data, scaler = preprocess_data(data)
    
        time_step = 60
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size, :]
        test_data = scaled_data[train_size:, :]
    
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
    
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
        model = build_model(time_step)
        model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1)
    
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
    
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform([y_train])
        y_test = scaler.inverse_transform([y_test])
    
        st.write('Model Training Complete!')
    
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(data.index, data['Close'], label='Actual Stock Price')
    
        train_index = data.index[time_step+1:train_size]
        test_index = data.index[train_size+time_step+1:]
    
        ax.plot(train_index, train_predict, label='Training Prediction')
        ax.plot(test_index, test_predict, label='Testing Prediction')
    
        ax.set_title('Stock Price Prediction using LSTM')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.legend()
    
        st.pyplot(fig)

if __name__ == "__main__":
    main()
