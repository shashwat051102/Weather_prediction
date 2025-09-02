import os
from src.Predict_weather import logger
from sklearn.model_selection import train_test_split
from src.Predict_weather.entity.config_entity import (DataTransformationConfig)
import pandas as pd




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    
    def transform_data(self):
        
        data = pd.read_csv(self.config.data_path)
        
        # Ensure the 'date' column exists before processing
        if 'date' not in data.columns:
            logger.error(f"Column 'date' not found in the dataset. Available columns: {list(data.columns)}")
            raise KeyError("Column 'date' not found in the dataset.")
        
        # Extract day, year, and month from the 'date' column
        data['day'] = pd.to_datetime(data['date'], errors='coerce').dt.day
        data['year'] = pd.to_datetime(data['date'], errors='coerce').dt.year
        data['month'] = pd.to_datetime(data['date'], errors='coerce').dt.month
        
        # Remove the original date column after extracting temporal features
        data = data.drop('date', axis=1)
        
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        
        logger.info("Splitted data into training and test sets")
        logger.info("Removed original date column after extracting temporal features")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")
        
        print(f"Train shape: {train.shape}")            
        print(f"Test shape: {test.shape}")
        print("Date column removed successfully!")