# app.py

import numpy as np
import pandas as pd
import requests
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="StockSense - Stock Price Predictor", layout="wide")
st.title("📈 StockSense - Stock Price Predictor")

# Dropdown for popular stocks
popular_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "INFY", "RELIANCE.BSE", "TCS.BSE"]
stock = st.selectbox("Choose a Stock Symbol", popular_stocks)

# Twelve Data API info
API_KEY = "aff4764122b9463aaaff6b18ee69550f"
BASE_URL = "https://api.twelvedata.com/time_series"

# Date range
start_date = "2005-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Fetch data
params = {
    "symbol": stock,
    "interval": "1day",
    "start_date": start_date,
    "end_date": end_date,
    "apikey": API_KEY,
    "format": "JSON",
    "outputsize": 5000
}

response = requests.get(BASE_URL, params=params)

if response.status_code != 200:
    st.error("❌ Failed to fetch data. Check the stock symbol or API key.")
    st.stop()

data_json = response.json()

if "values" not in data_json:
    st.error(f"❌ API Error: {data_json.get('message', 'No data returned')}")
    st.stop()

# Prepare DataFrame
data = pd.DataFrame(data_json["values"])
data["datetime"] = pd.to_datetime(data["datetime"])
data.set_index("datetime", inplace=True)
data.sort_index(inplace=True)
data = data.astype(float)

# Show raw data
with st.expander("📊 Show Raw Data"):
    st.write(data.tail())

# Split data
data_train = data[["close"]][:-100]
data_test = data[["close"]][-100:]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Load model
try:
    model = load_model("keras_model.h5")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# Prepare test input
past_100 = data_train.tail(100)
final_test = pd.concat([past_100, data_test], ignore_index=True)

if final_test.empty:
    st.error("❌ Not enough data to process.")
    st.stop()

input_data = scaler.transform(final_test)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_predicted = model.predict(x_test)

# Inverse scale
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting
st.subheader(f"📉 Predicted vs Actual Prices for {stock}")
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Actual Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
