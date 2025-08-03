import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set config before anything else
st.set_page_config(page_title="StockSense", layout="wide", page_icon="📈")

# App title
st.title("📈 StockSense - Predict Stock Prices using LSTM")

# Sidebar Inputs
st.sidebar.header("🔍 Stock Configuration")
ticker = st.sidebar.text_input("Ticker Symbol (e.g., AAPL, MSFT, GOOG)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))

# Internet and Host Check
st.subheader("🌐 Connectivity Check")
try:
    import socket
    socket.gethostbyname("finance.yahoo.com")
    st.success("✅ Internet and Yahoo Finance host accessible.")
except:
    st.error("❌ Cannot resolve Yahoo Finance host. Check DNS or Streamlit Cloud.")

# Download data
st.subheader("📊 Downloading Stock Data...")
try:
    import yfinance as yf
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("Empty data from yfinance, trying fallback...")
    st.success(f"✅ Data fetched from yfinance with {df.shape[0]} records.")
except Exception as e:
    st.warning(f"⚠️ yfinance failed: {e}")
    try:
        from yahooquery import Ticker as YQ_Ticker
        tq = YQ_Ticker(ticker)
        hist = tq.history(start=start_date, end=end_date)
        df = hist.reset_index().set_index("date")[["close"]].rename(columns={"close": "Close"})
        st.success(f"✅ Data fetched from yahooquery with {df.shape[0]} records.")
    except Exception as err:
        st.error(f"❌ Both yfinance and yahooquery failed: {err}")
        st.stop()

# Show raw data
st.write("📄 Raw Data Preview:")
st.dataframe(df.tail())

# Visualize historical price
st.subheader("📈 Historical Closing Price")
fig, ax = plt.subplots(figsize=(12, 4))
df['Close'].plot(ax=ax)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.title(f"{ticker} Stock Price")
st.pyplot(fig)

# Data preparation for LSTM
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df[['Close']])

train_size = int(len(data_scaled) * 0.80)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Predict
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test)

# Plot prediction
st.subheader("📉 Actual vs Predicted Prices")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(actual_prices, label="Actual")
ax2.plot(predicted_prices, label="Predicted")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.title(f"{ticker} Price Prediction")
st.pyplot(fig2)
