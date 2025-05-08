import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Define functions for loading data, training model, and making predictions
def load_data(ticker, period='6mo', interval='1h'):
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval)
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

def predict_next_24_hours(model, last_data, scaler):
    predictions = []
    directions = []
    input_data = last_data.copy()

    for _ in range(24):  # Predict for the next 24 hours
        pred = model.predict(input_data)
        pred_unscaled = scaler.inverse_transform(pred)
        predictions.append(pred_unscaled[0][0])

        # Determine direction: up (1) if the price is predicted to rise, down (0) otherwise
        last_price = scaler.inverse_transform(input_data[0])[-1][0]
        direction = "UP" if pred_unscaled[0][0] > last_price else "DOWN"
        directions.append(direction)

        # Update input data by appending the predicted price
        new_input = np.append(input_data[0][1:], pred, axis=0)
        input_data = new_input.reshape(1, -1, 1)

    return predictions, directions

def main():
    st.title('Next 24-Hour Stock Price and Direction Prediction')

    # Sidebar for ticker input
    ticker = st.sidebar.text_input('Enter Stock Ticker:', 'AMZN')
    period = st.sidebar.selectbox('Select Data Period:', ['1mo', '3mo', '6mo', '1y'])
    interval = st.sidebar.selectbox('Select Data Interval:', ['1m', '5m', '15m', '30m', '1h', '1d'])

    # Fetch and display current data
    data = load_data(ticker, period, interval)
    
    if data.empty:
        st.error("Could not retrieve data. Please check the ticker and try again.")
        return

    st.subheader(f'Current Prices for {ticker}')
    st.write(data.tail())  # Display last few records from the data for current prices
        
    # Preprocess data for model
    scaled_data, scaler = preprocess_data(data)
    time_step = 60
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size, :]
    test_data = scaled_data[train_size:, :]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build and train the LSTM model
    model = build_model(time_step)
    model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1)
    
    # Evaluate the model performance
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    st.subheader("Model Performance")
    st.write(f"Training MAE: {train_mae:.4f}")
    st.write(f"Test MAE: {test_mae:.4f}")
    st.write(f"Training RMSE: {train_rmse:.4f}")
    st.write(f"Test RMSE: {test_rmse:.4f}")
    
    # Next 24-hour predictions
    last_60_days = scaled_data[-time_step:]
    last_60_days = last_60_days.reshape(1, -1, 1)
    next_24_hours, next_24_directions = predict_next_24_hours(model, last_60_days, scaler)

    # Dataframe for next 24 hours
    future_times = pd.date_range(data.index[-1], periods=24, freq='H')
    prediction_df = pd.DataFrame({
        'Date/Time': future_times,
        'Predicted Price': next_24_hours,
        'Predicted Direction': next_24_directions
    })

    # Display next 24-hour predictions
    st.subheader(f'Predicted Stock Prices for the Next 24 Hours for {ticker}')
    st.dataframe(prediction_df)

    # Interactive Plotly Graph: Next 24-Hour Predicted Prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_times, y=next_24_hours, mode='lines', name='Next 24-Hour Predictions', line=dict(color='orange', dash='dash')))
    fig.update_layout(title=f'{ticker} Stock Price Prediction for Next 24 Hours', xaxis_title='Date/Time', yaxis_title='Stock Price')
    st.plotly_chart(fig)

    # Interactive Plotly Graph: Predicted Stock Direction
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=future_times, y=[1 if d == "UP" else 0 for d in next_24_directions], mode='markers+lines', marker=dict(color='green', size=8), name='Predicted Direction'))
    fig2.update_layout(title='Predicted Stock Direction (1: UP, 0: DOWN)', xaxis_title='Date/Time', yaxis_title='Direction', yaxis=dict(tickvals=[0, 1], ticktext=['DOWN', 'UP']))
    st.plotly_chart(fig2)

if __name__ == "__main__":
    main()
