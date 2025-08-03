import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

# Configure Streamlit
st.set_page_config(page_title="StockSense", layout="wide", page_icon="📈")
st.title("📈 StockSense - Predict Stock Prices using LSTM")

# Sidebar Configuration
st.sidebar.header("🔍 Stock Configuration")
stock_options = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOG)": "GOOG",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "NVIDIA (NVDA)": "NVDA"
}
stock_name = st.sidebar.selectbox("Choose Stock", list(stock_options.keys()))
ticker = stock_options[stock_name]

start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))

# Connectivity check
st.subheader("🌐 Connectivity Check")
try:
    import socket
    socket.gethostbyname("finance.yahoo.com")
    st.success("✅ Internet and Yahoo Finance host accessible.")
except:
    st.error("❌ Cannot resolve Yahoo Finance host.")

# Download data
st.subheader("📊 Downloading Stock Data...")
try:
    import yfinance as yf
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("Empty data from yfinance")
    st.success(f"✅ Data fetched with {df.shape[0]} records.")
except Exception as e:
    st.warning(f"⚠️ yfinance failed: {e}")
    try:
        from yahooquery import Ticker as YQ_Ticker
        tq = YQ_Ticker(ticker)
        hist = tq.history(start=start_date, end=end_date)
        df = hist.reset_index().set_index("date")[["close"]].rename(columns={"close": "Close"})
        st.success(f"✅ Data fetched from yahooquery with {df.shape[0]} records.")
    except Exception as err:
        st.error(f"❌ Both sources failed: {err}")
        st.stop()

# Show raw data
st.write("📄 Raw Data Preview:")
st.dataframe(df.tail())

# Plot historical closing price
st.subheader("📈 Historical Closing Price")
fig, ax = plt.subplots(figsize=(12, 4))
df['Close'].plot(ax=ax)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.title(f"{ticker} Stock Price")
st.pyplot(fig)

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Predict
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test)

# Prediction plot
st.subheader("📉 Actual vs Predicted Prices")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(actual_prices, label='Actual')
ax2.plot(predicted_prices, label='Predicted')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.title(f"{ticker} Price Prediction")
st.pyplot(fig2)

# Download buttons
st.subheader("📥 Download Results")

# CSV
csv_data = df.to_csv().encode()
st.download_button("Download CSV", csv_data, file_name=f"{ticker}_data.csv", mime='text/csv')

# PDF (convert matplotlib fig2 to PDF)
pdf_buffer = BytesIO()
fig2.savefig(pdf_buffer, format="pdf")
pdf_data = pdf_buffer.getvalue()
st.download_button("Download Prediction Plot (PDF)", pdf_data, file_name=f"{ticker}_prediction.pdf", mime="application/pdf")

# Footer
st.markdown("---")
st.markdown("""
### 👨‍💻 Made by Divyanshu  
📧 Email: [geekdivyxnsh@gmail.com](mailto:geekdivyxnsh@gmail.com)  
🔗 [GitHub](https://github.com/geekdivyxnsh) | [LinkedIn](https://www.linkedin.com/in/divyanshu-k-88a3a1266/)
""")
