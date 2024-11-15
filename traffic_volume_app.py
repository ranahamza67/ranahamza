import pandas as pd

# Load the dataset
df = pd.read_csv('TrafficVolumeData.csv')
df.head()
# Check for missing values
print(df.isnull().sum())
# Get basic statistics of the data
print(df.info())

print(df.describe())
# Check for duplicates
duplicates = df.duplicated()
print(duplicates)
import matplotlib.pyplot as plt

df['date_time'] = pd.to_datetime(df['date_time'])  # replace with your date column
df.set_index('date_time', inplace=True)
df['traffic_volume'].plot(figsize=(15, 6))
plt.title('Traffic Volume Over Time')
plt.show()
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
df.groupby('hour')['traffic_volume'].mean().plot()
plt.title('Average Traffic Volume by Hour')

plt.subplot(3, 1, 2)
df.groupby('day_of_week')['traffic_volume'].mean().plot()
plt.title('Average Traffic Volume by Day of the Week')

plt.subplot(3, 1, 3)
df.groupby('month')['traffic_volume'].mean().plot()
plt.title('Average Traffic Volume by Month')

plt.tight_layout()
plt.show()
def predict_traffic(input_date, input_hour):
    """
    Predict traffic volume for a given date and hour based on historical averages.
    
    Parameters:
        input_date (str): Date in 'YYYY-MM-DD' format.
        input_hour (int): Hour in 24-hour format (0-23).
    
    Returns:
        str: Predicted traffic volume or a message if no data is available.
    """
    try:
        # Convert input_date to pandas datetime
        input_date = pd.to_datetime(input_date).date()
        
        # Filter data for the specific date and hour
        df_filtered = df[(df.index.date == input_date) & (df.index.hour == input_hour)]
        
        if not df_filtered.empty:
            # Return average traffic volume for the hour
            return f"Predicted traffic volume: {df_filtered['traffic_volume'].mean():.2f}"
        else:
            return "No data available for the specified time."
    except Exception as e:
        return f"Error: {e}"

# Example usage:
date = '2016-01-01'  # Replace with your input date
hour = 14  # Replace with your input hour
print(predict_traffic(date, hour))
pip install streamlit
import streamlit as st
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('TrafficVolumeData.csv')
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month

# Pretrained Model (Random Forest as an example)
features = ['hour', 'day_of_week', 'month', 'air_pollution_index', 'humidity', 
            'wind_speed', 'visibility_in_miles', 'dew_point', 'temperature', 
            'rain_p_h', 'snow_p_h', 'clouds_all']
X = df[features]
y = df['traffic_volume']

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Streamlit Interface
st.title("Traffic Volume Predictor")
st.write("Provide details to predict traffic volume.")

# Inputs
input_date = st.date_input("Select Date", datetime.date(2016, 1, 1))
input_time = st.time_input("Select Time", datetime.time(12, 0))
air_pollution = st.slider("Air Pollution Index", 0, 500, 100)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 5)
visibility = st.slider("Visibility (miles)", 0, 10, 5)
dew_point = st.slider("Dew Point", -20, 30, 10)
temperature = st.slider("Temperature (°C)", -30, 50, 20)
rain_p_h = st.slider("Rain Probability (%)", 0, 100, 10)
snow_p_h = st.slider("Snow Probability (%)", 0, 100, 0)
clouds_all = st.slider("Cloud Cover (%)", 0, 100, 50)

# Prediction Logic
hour = input_time.hour
day_of_week = pd.Timestamp(input_date).dayofweek
month = pd.Timestamp(input_date).month

input_features = [hour, day_of_week, month, air_pollution, humidity, wind_speed, 
                  visibility, dew_point, temperature, rain_p_h, snow_p_h, clouds_all]

if st.button("Predict Traffic Volume"):
    prediction = model.predict([input_features])
    st.success(f"Predicted Traffic Volume: {prediction[0]:.2f}")

 