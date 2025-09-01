import os
import urllib.request as request
from src.Predict_weather import logger
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from src.Predict_weather.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
            retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
            openmeteo = openmeteo_requests.Client(session = retry_session)

            # Make sure all required weather variables are listed here
            # The order of variables in hourly or daily is important to assign them correctly below
            url = self.config.source_URL
            filename = self.config.local_data_file
            params = {
                "latitude": 22,
                "longitude": 79,
                "start_date": "2020-08-15",
                "end_date": "2025-08-15",
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "precipitation_hours", "temperature_2m_mean", "apparent_temperature_mean", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min", "sunshine_duration", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_gusts_10m_mean", "wind_speed_10m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean"],
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

            # Process daily data. The order of variables needs to be the same as requested.
            daily = response.Daily()
            daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
            daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
            daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
            daily_rain_sum = daily.Variables(3).ValuesAsNumpy()
            daily_precipitation_hours = daily.Variables(4).ValuesAsNumpy()
            daily_temperature_2m_mean = daily.Variables(5).ValuesAsNumpy()
            daily_apparent_temperature_mean = daily.Variables(6).ValuesAsNumpy()
            daily_cloud_cover_mean = daily.Variables(7).ValuesAsNumpy()
            daily_cloud_cover_max = daily.Variables(8).ValuesAsNumpy()
            daily_cloud_cover_min = daily.Variables(9).ValuesAsNumpy()
            daily_sunshine_duration = daily.Variables(10).ValuesAsNumpy()
            daily_wind_speed_10m_max = daily.Variables(11).ValuesAsNumpy()
            daily_wind_gusts_10m_max = daily.Variables(12).ValuesAsNumpy()
            daily_wind_gusts_10m_mean = daily.Variables(13).ValuesAsNumpy()
            daily_wind_speed_10m_mean = daily.Variables(14).ValuesAsNumpy()
            daily_relative_humidity_2m_max = daily.Variables(15).ValuesAsNumpy()
            daily_relative_humidity_2m_min = daily.Variables(16).ValuesAsNumpy()
            daily_relative_humidity_2m_mean = daily.Variables(17).ValuesAsNumpy()

            daily_data = {"date": pd.date_range(
                start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            )}

            daily_data["temperature_2m_max"] = daily_temperature_2m_max
            daily_data["temperature_2m_min"] = daily_temperature_2m_min
            daily_data["precipitation_sum"] = daily_precipitation_sum
            daily_data["rain_sum"] = daily_rain_sum
            daily_data["precipitation_hours"] = daily_precipitation_hours
            daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
            daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
            daily_data["cloud_cover_mean"] = daily_cloud_cover_mean
            daily_data["cloud_cover_max"] = daily_cloud_cover_max
            daily_data["cloud_cover_min"] = daily_cloud_cover_min
            daily_data["sunshine_duration"] = daily_sunshine_duration
            daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
            daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
            daily_data["wind_gusts_10m_mean"] = daily_wind_gusts_10m_mean
            daily_data["wind_speed_10m_mean"] = daily_wind_speed_10m_mean
            daily_data["relative_humidity_2m_max"] = daily_relative_humidity_2m_max
            daily_data["relative_humidity_2m_min"] = daily_relative_humidity_2m_min
            daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean

            daily_dataframe = pd.DataFrame(data = daily_data)
            print("\nDaily data\n", daily_dataframe)
            daily_dataframe.to_csv(filename, index = False)
            logger.info(f"Data ingestion completed. File saved at: {filename}")
        else:
            logger.info(f"File already exists at: {self.config.local_data_file}")