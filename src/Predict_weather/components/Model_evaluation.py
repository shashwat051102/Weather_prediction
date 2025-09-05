import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.keras
from src.Predict_weather.entity.config_entity import ModelEvaluationConfig
from src.Predict_weather.utils.common import save_json
from pathlib import Path
from tensorflow.keras.models import load_model
from src.Predict_weather import logger
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
    def create_sequences(self, data, seq_length):
        """Convert series to sliding windows for LSTM prediction"""
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        return rmse, mae, r2, mape
    
    def log_into_mlflow(self):
        
        test_data = pd.read_csv(self.config.test_data_path)
        logger.info(f"Test data shape: {test_data.shape}")
        
        model = load_model(self.config.model_path)
        scaler = joblib.load(self.config.scaler_path)
        
        target_column = self.config.target_column   
        target_data = test_data[[target_column]]
        
        scaled_data = scaler.transform(target_data)
        
        sequence_length = self.config.all_params.get("sequence_length", 60)
        X_test, y_test = self.create_sequences(scaled_data, sequence_length)
        
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        logger.info(f"Test sequences shape: {X_test.shape}, Test labels shape: {y_test.shape}")
        
        if self.config.mlflow_uri:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            
            predicted_scaled = model.predict(X_test)
            
            predicted_actual = scaler.inverse_transform(predicted_scaled)
            actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            predicted_actual = predicted_actual.flatten()
            actual_values = actual_values.flatten()
            
            rmse, mae, r2, mape = self.eval_metrics(actual_values, predicted_actual)
            
            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape
            }
            
            save_json(Path(self.config.metric_file_name), scores)
            
            mlflow.log_params(self.config.all_params)
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(model, "model", registered_model_name="LSTM")
            else:
                mlflow.keras.log_model(model, "model")
                
            mlflow.log_artifact(self.config.scaler_path, "preprocessing")
            
            # Log evaluation summary
            logger.info(f"Model Evaluation Completed!")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R² Score: {r2:.4f}")
            logger.info(f"MAPE: {mape:.2f}%")
            
            print(f"=== Model Evaluation Results ===")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"Test Samples: {len(actual_values)}")