from flask import Flask, render_template, request, jsonify, redirect, url_for
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import numpy as np
import joblib
# Don't import TensorFlow at startup - import when needed
import os
from datetime import datetime, timedelta

app = Flask(__name__)

class WeatherPredictor:
    def __init__(self, model_path="artifacts/model_trainer/model.h5", 
                 scaler_path="artifacts/model_trainer/scaler.pkl"):
        """Initialize the weather predictor with model and scaler paths."""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=300)  # was 3600, reduce to 5 minutes
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
    def load_model_and_scaler(self):
        """Load the trained model and scaler."""
        from tensorflow.keras.models import load_model
        
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
    
    def fetch_weather_data(self, latitude=20.5937, longitude=78.9629, days=65):
        """Fetch weather data using your exact OpenMeteo API code."""
        # Calculate date range for recent data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        # Your exact API code
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "weather_code", "cloud_cover", "cloud_cover_low", "rain", "surface_pressure", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m"],
            "timezone": "auto",
        }
        responses = self.openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation: {response.Elevation()} m asl")
        print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
        hourly_weather_code = hourly.Variables(5).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
        hourly_cloud_cover_low = hourly.Variables(7).ValuesAsNumpy()
        hourly_rain = hourly.Variables(8).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(9).ValuesAsNumpy()
        hourly_cloud_cover_mid = hourly.Variables(10).ValuesAsNumpy()
        hourly_cloud_cover_high = hourly.Variables(11).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(12).ValuesAsNumpy()
        hourly_wind_speed_100m = hourly.Variables(13).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(14).ValuesAsNumpy()
        hourly_wind_direction_100m = hourly.Variables(15).ValuesAsNumpy()
        hourly_wind_gusts_10m = hourly.Variables(16).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
        hourly_data["dew_point_2m"] = hourly_dew_point_2m
        hourly_data["apparent_temperature"] = hourly_apparent_temperature
        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["weather_code"] = hourly_weather_code
        hourly_data["cloud_cover"] = hourly_cloud_cover
        hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
        hourly_data["rain"] = hourly_rain
        hourly_data["surface_pressure"] = hourly_surface_pressure
        hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
        hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
        hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
        hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
        hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        print("\nhourly data\n", hourly_dataframe)
        
        return hourly_dataframe

    def fetch_current_temperature(self, latitude=20.5937, longitude=78.9629):
        """Fetch the current temperature and timestamp."""
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["temperature_2m"],
            "timezone": "auto",
        }
        responses = self.openmeteo.weather_api(url, params=params)
        response = responses[0]
        current = response.Current()
        current_temp = current.Variables(0).Value()
        current_time = pd.to_datetime(current.Time(), unit="s", utc=True)
        return current_time, float(current_temp)
    
    def prepare_features(self, df):
        """Prepare scaled temperature series for LSTM input.
        The scaler in training was fit on the target column only (temperature_2m),
        so we must scale only that column and pass numpy arrays (not DataFrames)
        to avoid scikit-learn feature name checks.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not loaded")
        if 'temperature_2m' not in df.columns:
            raise ValueError("temperature_2m column missing in fetched data")
        temps = df['temperature_2m'].astype(float).values.reshape(-1, 1)
        temps_scaled = self.scaler.transform(temps)
        return temps_scaled
    
    def predict_temperature(self, latitude=20.5937, longitude=78.9629, sequence_length=60):
        """Predict the next day's temperature using last `sequence_length` hours."""
        # Fetch recent weather data
        df = self.fetch_weather_data(latitude, longitude, days=sequence_length + 10)
        
        # Prepare scaled temperature series
        temps_scaled = self.prepare_features(df)
        
        if temps_scaled.shape[0] < sequence_length:
            raise ValueError(f"Not enough data points: got {temps_scaled.shape[0]}, need {sequence_length}")
        
        # Build the last sequence
        sequence = temps_scaled[-sequence_length:]
        X = sequence.reshape(1, sequence_length, 1)
        
        # Predict scaled temperature
        prediction_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform the single predicted value
        pred_actual = self.scaler.inverse_transform(np.array([[prediction_scaled[0, 0]]]))[0, 0]
        return float(pred_actual)

# Initialize the predictor (but don't load model yet)
predictor = WeatherPredictor()

def get_predictor():
    """Lazy load the predictor to avoid blocking startup."""
    if predictor.model is None or predictor.scaler is None:
        predictor.load_model_and_scaler()
    return predictor

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Weather Prediction API is running'})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests."""
    try:
        # If user visits /predict directly, redirect to form on home page
        if request.method == 'GET':
            return redirect(url_for('index'))
        
        # Get latitude and longitude from form
        latitude = float(request.form.get('latitude', 20.5937))
        longitude = float(request.form.get('longitude', 78.9629))
        
        # Make prediction using lazy-loaded predictor
        pred = get_predictor()
        predicted_temp = pred.predict_temperature(latitude, longitude)
        
        # Fetch recent history for chart (previous 10 hours only)
        history_df = pred.fetch_weather_data(latitude, longitude, days=2)
        last10 = history_df.tail(10)
        chart_labels = last10['date'].astype(str).tolist()
        chart_values = last10['temperature_2m'].astype(float).round(2).tolist()
        
        return render_template('result.html', 
                             prediction=f"{predicted_temp:.2f}°C",
                             latitude=latitude,
                             longitude=longitude,
                             chart_labels=chart_labels,
                             chart_values=chart_values)
    
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        return render_template('result.html', error=error_message)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()
        latitude = float(data.get('latitude', 20.5937))
        longitude = float(data.get('longitude', 78.9629))
        
        pred = get_predictor()
        predicted_temp = pred.predict_temperature(latitude, longitude)
        
        # Include last 10 hours only for clients that wish to plot
        history_df = pred.fetch_weather_data(latitude, longitude, days=2)
        last10 = history_df.tail(10)
        chart_labels = last10['date'].astype(str).tolist()
        chart_values = last10['temperature_2m'].astype(float).round(2).tolist()
        
        return jsonify({
            'success': True,
            'prediction': f"{predicted_temp:.2f}°C",
            'latitude': latitude,
            'longitude': longitude,
            'history': {
                'labels': chart_labels,
                'values': chart_values
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("Starting Flask Weather Prediction App...")
    print("Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)