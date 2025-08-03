# app.py

import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Set Streamlit config
st.set_page_config(page_title="StockSense", layout="wide", page_icon="📈")

# Sidebar
st.sidebar.title("🧭 Select Stock")
stock = st.sidebar.selectbox(
    "Choose a stock to predict:",
    options=["GOOG", "AAPL", "MSFT", "AMZN", "TSLA", "META", "NFLX"],
    index=0
)

# App title
st.title("📈 StockSense - Stock Price Predictor")
st.markdown("---")

# Define date range
start_date = '2010-01-01'
end_date = '2024-12-31'

# Download data
st.subheader("⏳ Fetching Stock Data...")
try:
    data = yf.download(stock, start=start_date, end=end_date)
except Exception as e:
    st.error(f"Failed to download data: {e}")
    st.stop()

# Validate data
if data.empty:
    st.error("Downloaded data is empty. Please check ticker symbol or date range.")
    st.stop()

st.write("📅 Date Range:", f"{start_date} to {end_date}")
st.write("📊 Raw data preview:")
st.dataframe(data.tail(), use_container_width=True)

# Moving averages
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

st.subheader("📉 100 & 200 Day Moving Averages")
fig1 = plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Original', color='green')
plt.plot(ma_100, label='100-Day MA', color='red')
plt.plot(ma_200, label='200-Day MA', color='blue')
plt.title(f"{stock} Stock Price with Moving Averages")
plt.legend()
st.pyplot(fig1)

# Preprocessing
data = data[['Close']].dropna()
data_train = data[:int(len(data)*0.8)]
data_test = data[int(len(data)*0.8):]

# Check and scale data
if data_train.empty:
    st.error("Training data is empty. Cannot proceed with model.")
    st.stop()

scaler = MinMaxScaler()
data_train_scaled = scaler.fit_transform(data_train)

x_train, y_train = [], []
for i in range(100, len(data_train_scaled)):
    x_train.append(data_train_scaled[i-100:i])
    y_train.append(data_train_scaled[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Load trained model
try:
    model = load_model('Stock_Predictions_Model.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prepare test data
past_100 = data_train.tail(100)
final_test = pd.concat([past_100, data_test], ignore_index=True)
final_test_scaled = scaler.transform(final_test)

x_test, y_test = [], []
for i in range(100, len(final_test_scaled)):
    x_test.append(final_test_scaled[i-100:i])
    y_test.append(final_test_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Predict
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
st.subheader("📈 Actual vs Predicted Stock Price")
fig2 = plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Price', color='green')
plt.plot(y_pred, label='Predicted Price', color='red')
plt.xlabel("Time")
plt.ylabel(f"{stock} Price")
plt.title(f"{stock} Stock Price Prediction")
plt.legend()
st.pyplot(fig2)

# Download prediction results
st.subheader("⬇️ Download Predictions")
pred_df = pd.DataFrame({
    "Actual Price": y_test.flatten(),
    "Predicted Price": y_pred.flatten()
})

csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("Download as CSV", csv, f"{stock}_prediction.csv", "text/csv")

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    pred_df.to_excel(writer, index=False, sheet_name="Predictions")

st.download_button(
    "Download as Excel",
    excel_buffer.getvalue(),
    file_name=f"{stock}_prediction.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Footer
st.markdown("""
---
<div style='text-align: center; color: gray;'>
    <strong>Made by Divyanshu</strong><br>
    📧 geekdivyxnsh@gmail.com<br>
    🔗 <a href='https://www.linkedin.com/in/divyanshu-k-88a3a1266/' target='_blank' style='color: lightblue;'>LinkedIn</a> |
    <a href='https://github.com/geekdivyxnsh' target='_blank' style='color: lightblue;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
