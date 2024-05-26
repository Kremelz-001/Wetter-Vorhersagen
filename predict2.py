import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Reading the data
weather = pd.read_csv("weather.csv", index_col="DATE")

def fahrtocel(f):
    return (f - 32) * (5/9)

# Filtering the data for training the model
null_pct = weather.isnull().mean()
valid_columns = null_pct[null_pct < 0.04].index
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather = weather.ffill()

# Convert Index into the datatype as date, remove gaps from data
weather.index = pd.to_datetime(weather.index)

# Predicting max temp
weather["target"] = weather.shift(-1)["tmax"]
weather = weather.ffill()

# Initialize Ridge regression model
rr = Ridge(alpha=0.2)

# List of columns we wish to predict
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]

# Train the model on historical data
rr.fit(weather[predictors], weather["target"])

# Define a function to predict future weather
def predict_future(weather, model, predictors, start_date, end_date):
    # Generate future dates
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_weather = pd.DataFrame(index=future_dates, columns=predictors)
    
    # Predict for each future date
    for date in future_dates:
        # Use the most recent data available for prediction
        recent_data = weather[weather.index <= date].iloc[-1]
        # Predict using the model
        prediction = model.predict([recent_data[predictors]])
        # Store the prediction for the current date
        future_weather.loc[date] = prediction
    
    return future_weather

# Define the start and end dates for future prediction
start_date = weather.index[-1] + pd.Timedelta(days=1)
end_date = start_date + pd.Timedelta(days=30)  # Predict for the next 30 days

# Predict future weather
future_weather = predict_future(weather, rr, predictors, start_date, end_date)

# Plot the predicted temperatures
plt.plot(future_weather.index, future_weather["tmax"], label="Predicted Max Temp")
plt.plot(weather.index, weather["tmax"], label="Historical Max Temp")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Predicted Max Temperature for the Next 30 Days")
plt.legend()
plt.show()









