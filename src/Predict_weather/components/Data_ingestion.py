import os
import urllib.request as request
from src.Predict_weather import logger
# import openmeteo_requests  # Moved to conditional import
import pandas as pd
# import requests_cache  # Moved to conditional import
# from retry_requests import retry  # Moved to conditional import
from src.Predict_weather.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            # Import only when needed to avoid import errors when file exists
            import openmeteo_requests
            import requests_cache
            from retry_requests import retry
            
            cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
            retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
            openmeteo = openmeteo_requests.Client(session = retry_session)

            # Make sure all required weather variables are listed here
            # The order of variables in hourly or hourly is important to assign them correctly below
            url = self.config.source_URL
            filename = self.config.local_data_file
            params = {
            "latitude": 22,
            "longitude": 79,
            "start_date": "2024-08-19",
            "end_date": "2025-09-02",
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "weather_code", "cloud_cover", "cloud_cover_low", "rain", "surface_pressure", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m"],
            "timezone": "auto",
            "utm_source": "chatgpt.com",
            }
            responses = openmeteo.weather_api(url, params=params)

            # Process first location. Add a for-loop for multiple locations or weather models
            response = responses[0]
            print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
            print(f"Elevation: {response.Elevation()} m asl")
            print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
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
            hourly_dataframe.to_csv(filename, index = False)
            logger.info(f"Data ingestion completed. File saved at: {filename}")
        else:
            logger.info(f"File already exists at: {self.config.local_data_file}")