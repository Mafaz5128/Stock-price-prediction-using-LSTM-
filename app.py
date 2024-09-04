import streamlit as st
def main():
    st.title('AMZN Stock Price Prediction with LSTM')
    
    ticker = st.text_input('Enter Stock Ticker:', 'AMZN')
    data = load_data(ticker)
    
    if data.empty:
        st.warning("No data available. Please check the ticker symbol.")
        return
    
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
