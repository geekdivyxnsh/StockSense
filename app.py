# app.py

import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="StockSense", page_icon="📈")

# Custom dark theme
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f0f0f, #1e1e1e);
        color: white;
    }
    .stApp {
        background-color: #121212;
    }
    .css-1d391kg, .css-1v3fvcr, .css-1cpxqw2 {
        color: white !important;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for stock selection
st.sidebar.title("🧭 Select Stock")
stock = st.sidebar.selectbox(
    "Choose a stock to predict:",
    options=["GOOG", "AAPL", "MSFT", "AMZN", "TSLA", "META", "NFLX"],
    index=0
)

# App Title
st.title("📈 StockSense - Stock Price Predictor")
st.markdown("""---""")

# Date Range
start = '2005-01-01'
end = '2025-06-30'

# Download stock data
data = yf.download(stock, start=start, end=end, auto_adjust=False)
data.reset_index(inplace=True)

# Display raw data
st.subheader("🔍 Historical Stock Data")
st.dataframe(data.tail(), use_container_width=True)

# Moving Averages
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

st.subheader("📊 Moving Averages")
fig1 = plt.figure(figsize=(9,5))  # Reduced width by 1 inch
plt.plot(ma_100, 'r', label='100-Day MA')
plt.plot(ma_200, 'b', label='200-Day MA')
plt.plot(data.Close, 'g', label='Original')
plt.legend()
plt.title(f"{stock} Stock with MA100 & MA200")
st.pyplot(fig1)

# Preprocessing
data.dropna(inplace=True)
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

scaler = MinMaxScaler(feature_range=(0,1))
data_train_scale = scaler.fit_transform(data_train)

x = []
y = []
for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i, 0])

x, y = np.array(x), np.array(y)
x = x.reshape(x.shape[0], x.shape[1], 1)

# Load model
model = load_model('Stock_Predictions_Model.keras')

# Prepare test data
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

x_test = []
y_test = []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Prediction
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# Final Plot
st.subheader("📉 Actual vs Predicted Prices")
fig2 = plt.figure(figsize=(9,6))  # Reduced width by 1 inch
plt.plot(y_pred, 'r', label='Predicted Price')
plt.plot(y_test, 'g', label='Actual Price')
plt.xlabel('Time')
plt.ylabel(f'{stock} Price')
plt.title(f'{stock} Price Prediction')
plt.legend()
st.pyplot(fig2)

# Download CSV/Excel
st.subheader("⬇️ Download Prediction Results")
pred_df = pd.DataFrame({
    'Actual Price': y_test.flatten(),
    'Predicted Price': y_pred.flatten()
})

# CSV
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name=f'{stock}_prediction.csv', mime='text/csv')

# Excel
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    pred_df.to_excel(writer, index=False, sheet_name='Sheet1')

st.download_button(
    label="Download Excel",
    data=excel_buffer.getvalue(),
    file_name=f'{stock}_prediction.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# Footer
st.markdown("""
---
<div style='text-align: center; color: gray;'>
    <strong>Made by Divyanshu</strong><br>
    📧 geekdivyxnsh@gmail.com<br>
    🔗 <a href='https://www.linkedin.com/in/divyanshu-k-88a3a1266/' style='color: lightblue;' target='_blank'>LinkedIn</a> |
    <a href='https://github.com/geekdivyxnsh' style='color: lightblue;' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)
